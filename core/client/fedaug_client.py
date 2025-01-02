import os
import copy
import logging

import ray

import torch
from torch.utils.data import Dataset, DataLoader

from core.client.base_client import BaseClient
from utils.utils import initLogging, seed_anything

class UserItemRatingDataset(Dataset):
    def __init__(self, user_tensor, item_tensor, rating_tensor):
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.rating_tensor = rating_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.rating_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)

class UserItemSeqDataset(Dataset):
    def __init__(self, uid, seq, max_length=5, stride=1, condational_method="add"):

        self.input_ids = []
        self.target_ids = []
        self.user_ids = []

        for i in range(0, len(seq) - max_length, stride):
            input_chunk = seq[i:i + max_length]

            if condational_method == 'add':
                target_chunk = seq[i + 1:i + max_length + 1]
            elif condational_method == 'concat':
                target_chunk = seq[i:i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
            # self.user_ids.append(torch.tensor([uid] * len(input_chunk)))
            self.user_ids.append(torch.tensor([uid]))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx], self.user_ids[idx]

@ray.remote(num_gpus=0.25)
class FedAugActor(BaseClient):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.device = torch.device(self.args['device'])
        seed_anything(args['seed'])
        initLogging(args['log_dir'] / f"client_{os.getpid()}.log", stream=False)

    def train_decoder(self, decoder, user_data):
        client_decoder = copy.deepcopy(decoder)
        user, train_seq_data  = user_data[0], user_data[1]['train_positive']
        # if user['user_id'] == 0:
        #     print(user['user_id'])
        #     print(len(train_seq_data))
        # print(user['decoder_dict'].keys())
        if user['decoder_dict'] is not None:
            user_decoder_dict = client_decoder.state_dict() | user['decoder_dict']
            client_decoder.load_state_dict(user_decoder_dict)

        client_decoder = client_decoder.to(self.device)

        optimizer = torch.optim.Adam(client_decoder.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

        seq_dataloader = DataLoader(
            dataset=UserItemSeqDataset(uid=user['user_id'], seq=train_seq_data),
            batch_size=self.args['batch_size'],
            shuffle=True
        )

        client_decoder.train()
        client_decoder_loss = []
        client_decoder_acc = []
        for epoch in range(self.args['local_epoch']):
            epoch_loss, samples = 0, 0
            total_correct_predictions = 0
            loss_fn = torch.nn.CrossEntropyLoss()

            for input_ids, target_ids, user_ids in seq_dataloader:
                input_ids, target_ids, user_ids = input_ids.to(self.device), target_ids.to(self.device), user_ids.to(self.device)

                optimizer.zero_grad()
                outputs = client_decoder(input_ids, user_ids)
                logits = outputs.logits

                # Compute predictions and accuracy
                _, predicted = torch.max(logits, dim=-1)
                correct_predictions = (predicted.view(-1) == target_ids.view(-1)).sum().item()
                total_correct_predictions += correct_predictions

                # Compute loss
                loss = loss_fn(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                loss.backward()

                optimizer.step()
                scheduler.step()

                # epoch_loss += loss.item() * len(user_ids)
                # samples += len(user_ids)
                epoch_loss += loss.item() * len(user_ids) * target_ids.size(1)
                samples += len(user_ids) * target_ids.size(1)

            client_decoder_loss.append(epoch_loss / samples)
            client_decoder_acc.append(total_correct_predictions / samples)
        
        client_decoder.to('cpu')
        logging.info(f"User {user['user_id']} training decoder finished with loss = {client_decoder_loss}.")
        logging.info(f"User {user['user_id']} training decoder finished with train_acc = {client_decoder_acc}.")
        return user['user_id'], client_decoder, client_decoder_loss

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