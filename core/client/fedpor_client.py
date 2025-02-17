import os
import copy
import logging

import ray

import torch

from core.client.base_client import BaseClient
from dataset.base_dataset import UserItemRatingDataset
from utils.utils import initLogging, seed_anything

@ray.remote
class FedPORActor(BaseClient):
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

        if self.args['optimizer'] == "SGD":
            optimizer = torch.optim.SGD([
                {'params': user_model.user_embedding.parameters(), 'lr': self.args['lr_network']},
                {'params': user_model.item_personality.parameters(), 'lr': self.args['lr_args']},
                {'params': user_model.item_commonality.parameters(), 'lr': self.args['lr_args']},
            ], weight_decay=self.args['l2_regularization'])

        elif self.args['optimizer'] == "Adam":
            optimizer = torch.optim.Adam([
                {'params': user_model.user_embedding.parameters(), 'lr': self.args['lr_network']},
                {'params': user_model.item_personality.parameters(), 'lr': self.args['lr_args']},
                {'params': user_model.item_commonality.parameters(), 'lr': self.args['lr_args']},
            ], weight_decay=self.args['l2_regularization'])
        else:
            raise NotImplementedError(f"Not implemented optimizer: {self.args['optimizer']}")

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

        dataloader = torch.utils.data.DataLoader(
            dataset=UserItemRatingDataset(
                    user_tensor=torch.LongTensor(train_data[0]),
                    item_tensor=torch.LongTensor(train_data[1]),
                    rating_tensor=torch.LongTensor(train_data[2])),
            batch_size=self.args['batch_size'], shuffle=True
        )

        user_model.train()
        training_loss = []
        for epoch in range(self.args['local_epoch']):
            epoch_loss, samples = 0, 0
            for users, items, ratings in dataloader:
                loss_fn = torch.nn.BCELoss()

                users, items, ratings = users.to(self.device), items.to(self.device), ratings.float().to(self.device)

                optimizer.zero_grad()
                ratings_pred, _, _ = user_model(items)

                loss = loss_fn(ratings_pred.view(-1), ratings)
                loss.backward()
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item() * len(users)
                samples += len(users)
            training_loss.append(epoch_loss / samples)

            # check convergence
            try:
                if epoch > 0 and abs(training_loss[epoch] - training_loss[epoch - 1]) / abs(
                        training_loss[epoch - 1]) < self.args['tol']:
                    break
            except ZeroDivisionError:
                logging.info(f"epoch={epoch} = {training_loss[epoch]}, last epoch = {training_loss[epoch - 1]} ")

        user_model.to('cpu')
        return user_context['user_id'], user_model, training_loss
