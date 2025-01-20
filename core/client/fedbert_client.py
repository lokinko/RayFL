import os
import copy
import logging
import random
import ray

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from core.client.base_client import BaseClient
from utils.utils import initLogging, seed_anything
class UserBertTrainDataset(Dataset):
    def __init__(self, user_seq, max_len, mask_prob, mask_token, num_items, seed):
        self.user_seq = user_seq
        self.user_seqs = [user_seq for _ in range(2)]
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.num_items = num_items
        self.rng = random.Random(seed)
        self.tokens_list = []
        self.labels_list = []

        for seq in self.user_seqs:
            tokens = []
            labels = []
            for s in seq:
                prob = self.rng.random()
                if prob < self.mask_prob:
                    prob /= self.mask_prob

                    if prob < 0.8:
                        tokens.append(self.mask_token)
                    elif prob < 0.9:
                        tokens.append(self.rng.randint(1, self.num_items))
                    else:
                        tokens.append(s)

                    labels.append(s)
                else:
                    tokens.append(s)
                    labels.append(0)

            tokens = tokens[-self.max_len:]
            labels = labels[-self.max_len:]

            mask_len = self.max_len - len(tokens)

            tokens = [0] * mask_len + tokens
            labels = [0] * mask_len + labels

            labels = [int(l) for l in labels]
            self.tokens_list.append(torch.tensor(tokens))
            self.labels_list.append(torch.tensor(labels))

    def __len__(self):
        return len(self.user_seqs)

    def __getitem__(self, index):
        return self.tokens_list[index], self.labels_list[index]


# class UserBertTrainDataset(Dataset):
#     def __init__(self, user_seq, max_len, mask_prob, mask_token, num_items):
#         """
#         初始化单用户的数据集。
        
#         参数:
#         - user_seq: list[int], 单个用户的序列。
#         - max_len: int, 最大序列长度。
#         - mask_prob: float, 被mask的概率。
#         - mask_token: int, 用于表示mask的token ID。
#         - num_items: int, 所有可能item的数量（用于随机替换）。
#         - rng: random.Random, 随机数生成器对象。
#         """
#         self.user_seq = user_seq
#         self.max_len = max_len
#         self.mask_prob = mask_prob
#         self.mask_token = mask_token
#         self.num_items = num_items
#         self.rng = random.Random(0)
#         self.repeat = 10
#     def __len__(self):
#         return self.repeat
#     def __getitem__(self, index):
#         seq = self.user_seq

#         tokens = []
#         labels = []
#         for s in seq:
#             prob = self.rng.random()
#             if prob < self.mask_prob:
#                 prob /= self.mask_prob

#                 if prob < 0.8:
#                     tokens.append(self.mask_token)
#                 elif prob < 0.9:
#                     tokens.append(self.rng.randint(1, self.num_items))
#                 else:
#                     tokens.append(s)

#                 labels.append(s)
#             else:
#                 tokens.append(s)
#                 labels.append(0)

#         tokens = tokens[-self.max_len:]
#         labels = labels[-self.max_len:]

#         mask_len = self.max_len - len(tokens)

#         tokens = [0] * mask_len + tokens
#         labels = [0] * mask_len + labels

#         # 确保所有元素都是整数
#         tokens = [int(t) for t in tokens]
#         labels = [int(l) for l in labels]
#         # print(labels)
#         return torch.LongTensor(tokens), torch.LongTensor(labels)

@ray.remote(num_gpus=0.125)
class FedBertActor(BaseClient):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.device = torch.device(self.args['device'])
        seed_anything(args['seed'])
        initLogging(args['log_dir'] / f"client_{os.getpid()}.log", stream=False)
        self.ce = torch.nn.CrossEntropyLoss(ignore_index=0)

        self.num_items = args['num_items']
        print(self.num_items)

    def calculate_loss(self, encoder, batch):
        
        seqs, labels = batch # seqs  B x T [2, 100]
        logits = encoder(seqs)  # B x T x V

        logits = logits.view(-1, logits.size(-1))  # (B*T) x V
        labels = labels.view(-1)  # B*T
        loss = self.ce(logits, labels)

        # 获取top-10的索引
        _, top_k_indices = torch.topk(logits, k=10, dim=1)

        # 找到非零label的mask
        non_zero_mask = labels > 0

        # 只对非零label进行处理
        non_zero_labels = labels[non_zero_mask].unsqueeze(-1)  # [num_non_zero, 1]
        relevant_top_k_indices = top_k_indices[non_zero_mask]  # [num_non_zero, 10]

        # 利用广播机制比较
        matches = (relevant_top_k_indices == non_zero_labels).any(dim=1)

        # 计算准确率
        accuracy = matches.float().mean().item()
        # print(logits.shape)
        # print(labels.shape)
        # print(labels.count_nonzero().item())
        return loss, accuracy

    def train_encoder(self, encoder, user_data):
        client_encoder = copy.deepcopy(encoder)
        user, train_seq_data = user_data[0], user_data[1]['train_positive']

        if user['encoder_dict'] is not None:
            user_encoder_dict = client_encoder.state_dict() | user['encoder_dict']
            client_encoder.load_state_dict(user_encoder_dict)

        client_encoder = client_encoder.to(self.device)

        optimizer = torch.optim.Adam(client_encoder.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

        seed = random.randint(0, 1000)
        dataloader = DataLoader(
            dataset=UserBertTrainDataset(user_seq=train_seq_data, 
                                        max_len=100, 
                                        mask_prob=0.3, 
                                        mask_token=self.num_items+1, 
                                        num_items=self.num_items,
                                        seed=seed),
            batch_size=self.args['batch_size'],
            shuffle=True
        )

        client_encoder.train()
        client_encoder_loss = []
        client_encoder_acc = []
        
        for epoch in range(self.args['local_epoch']):
            epoch_loss, samples = 0, 0
            
            for batch_idx, batch in enumerate(dataloader):
                batch_size = batch[0].size(0)
                batch = [x.to(self.device) for x in batch]

                optimizer.zero_grad()
                loss, acc = self.calculate_loss(client_encoder, batch)
                # if epoch == 0:
                #     logging.info(f"User {user['user_id']} seq {batch[0]}.")
                #     logging.info(f"User {user['user_id']} label {batch[1]}.")
                loss.backward()
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item() * batch_size
                if np.isnan(epoch_loss):
                    print(batch)
                samples += batch_size
            client_encoder_loss.append(epoch_loss / samples)
            client_encoder_acc.append(acc)
            # # check convergence
            # if epoch > 0 and client_encoder_loss[epoch] > client_encoder_loss[epoch - 1]:
            #     break
            # if epoch > 0 and abs(client_encoder_loss[epoch] - client_encoder_loss[epoch - 1]) / abs(
            #         client_encoder_loss[epoch - 1]) < self.args['tol']:
            #     break

        client_encoder.to('cpu')
        logging.info(f"User {user['user_id']} training finished with loss = {client_encoder_loss}.")
        return user['user_id'], client_encoder, client_encoder_loss, client_encoder_acc
    
    def train(self, model, user_data):
        client_model = copy.deepcopy(model)
        user, train_data = user_data[0], user_data[1]['train']

        if user['model_dict'] is not None:
            user_model_dict = client_model.state_dict() | user['model_dict']
            client_model.load_state_dict(user_model_dict)

        client_model = client_model.to(self.device)

        optimizer = torch.optim.Adam([
                {'params': client_model.user_embedding.parameters(), 'lr': self.args['lr_network']},
                {'params': client_model.item_commonality.parameters(), 'lr': self.args['lr_args']},
            ])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

        dataloader = DataLoader(
            dataset=UserItemRatingDataset(
                    user_tensor=torch.LongTensor(train_data[0]),
                    item_tensor=torch.LongTensor(train_data[1]),
                    rating_tensor=torch.LongTensor(train_data[2])),
            batch_size=self.args['batch_size'],
            shuffle=True
        )

        client_model.train()
        client_loss = []
        for epoch in range(self.args['local_epoch']):
            epoch_loss, samples = 0, 0
            loss_fn = torch.nn.BCELoss()
            for users, items, ratings in dataloader:
                users, items, ratings = users.to(self.device), items.to(self.device), ratings.float().to(self.device)
                optimizer.zero_grad()
                ratings_pred, items_commonality = client_model(items)

                loss = loss_fn(ratings_pred.view(-1), ratings)
                loss.backward()
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item() * len(users)
                samples += len(users)
            client_loss.append(epoch_loss / samples)

            # check convergence
            if epoch > 0 and abs(client_loss[epoch] - client_loss[epoch - 1]) / abs(
                    client_loss[epoch - 1]) < self.args['tol']:
                break

        client_model.to('cpu')
        logging.info(f"User {user['user_id']} training finished with loss = {client_loss}.")
        return user['user_id'], client_model, client_loss