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


class FedRapLoss(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.crit = torch.nn.BCELoss()
        self.independency = torch.nn.MSELoss()

        if self.args['regular'] == 'l2':
            self.reg = torch.nn.MSELoss()
        elif self.args['regular'] == 'l1':
            self.reg = torch.nn.L1Loss()
        else:
            self.reg = torch.nn.MSELoss()

    def forward(self, ratings_pred, ratings, item_personality, item_commonality):
        if self.args['regular'] == 'l2':
            dummy_target = torch.zeros_like(item_commonality, requires_grad=False)
            third = self.reg(item_commonality, dummy_target)
        elif self.args['regular'] == 'l1':
            dummy_target = torch.zeros_like(item_commonality, requires_grad=False)
            third = self.reg(item_commonality, dummy_target)
        elif self.args['regular'] == 'none':
            self.args['mu'] = 0
            dummy_target = item_commonality
            third = self.reg(item_commonality, dummy_target)
        elif self.args['regular'] == 'nuc':
            third = torch.norm(item_commonality, p='nuc')
        elif self.args['regular'] == 'inf':
            third = torch.norm(item_commonality, p=float('inf'))
        else:
            dummy_target = torch.zeros_like(item_commonality, requires_grad=False)
            third = self.reg(item_commonality, dummy_target)

        loss = self.crit(ratings_pred, ratings) \
               - self.args['lambda'] * self.independency(item_personality, item_commonality) \
               + self.args['mu'] * third

        return loss


@ray.remote
class FedRapActor(BaseClient):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.device = torch.device(self.args['device'])
        seed_anything(args['seed'])
        initLogging(args['log_dir'] / f"client_{os.getpid()}.log", stream=False)

    def train(self, model, user_data):
        user_context, train_data = user_data[0], user_data[1]

        user_model = copy.deepcopy(model)
        user_model.load_state_dict(user_context['state_dict'])

        user_model = user_model.to(self.device)
        optimizer = torch.optim.Adam([
            {'params': user_model.user_embedding.parameters(), 'lr': self.args['lr_network']},
            {'params': user_model.item_personality.parameters(), 'lr': self.args['lr_args']},
            {'params': user_model.item_commonality.parameters(), 'lr': self.args['lr_args']},
        ], weight_decay=self.args['l2_regularization'])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

        dataloader = DataLoader(
            dataset=UserItemRatingDataset(
                    user_tensor=torch.LongTensor(train_data[0]),
                    item_tensor=torch.LongTensor(train_data[1]),
                    rating_tensor=torch.LongTensor(train_data[2])),
            batch_size=self.args['batch_size'],
            shuffle=True
        )

        user_model.train()
        training_loss = []
        for epoch in range(self.args['local_epoch']):
            epoch_loss, samples = 0, 0
            loss_fn = FedRapLoss(self.args)
            for users, items, ratings in dataloader:
                users, items, ratings = users.to(self.device), items.to(self.device), ratings.float().to(self.device)
                optimizer.zero_grad()
                ratings_pred, items_personality, items_commonality = user_model(items)
                logging.info(
                    f"ratings_pred: {ratings_pred}, items_personality: {items_personality}, items_commonality: {items_commonality}")

                loss = loss_fn(ratings_pred.view(-1), ratings, items_personality, items_commonality)
                loss.backward()
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item() * len(users)
                samples += len(users)
            training_loss.append(epoch_loss / samples)

            # check convergence
            if epoch > 0 and abs(training_loss[epoch] - training_loss[epoch - 1]) / abs(
                    training_loss[epoch - 1]) < self.args['tol']:
                break

        user_model.to('cpu')
        return user_context['user_id'], user_model, training_loss
