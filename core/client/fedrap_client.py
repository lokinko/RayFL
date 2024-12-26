import copy

import ray

import torch
from torch.utils.data import Dataset, DataLoader

from core.client.base_client import BaseClient

class UserItemRatingDataset(Dataset):
    def __init__(self, user_tensor, item_tensor, rating_tensor):
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.rating_tensor = rating_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.rating_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)

@ray.remote
class FedRapActor(BaseClient):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.device = torch.device(self.args['device'])

    def train(self, train_data, model, user_model_dict=None):
        client_model = copy.deepcopy(model)
        if user_model_dict is not None:
            user_model_dict = client_model.state_dict() | user_model_dict
            client_model.load_state_dict(user_model_dict)

        client_model.to(self.args['device'])
        optimizer = torch.optim.SGD([
            {'params': client_model.affine_output.parameters(), 'lr': self.config['lr_network']},
            {'params': client_model.item_personality.parameters(), 'lr': self.config['lr_args']},
            {'params': client_model.item_commonality.parameters(), 'lr': self.config['lr_args']},
        ], weight_decay=self.config['l2_regularization'])
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
            for user, item, rating in dataloader:
                user, item, rating = user.to(self.device), item.to(self.device), rating.float().to(self.device)
                optimizer.zero_grad()
                ratings_pred, item_personality, item_commonality = client_model(item)

                loss = torch.nn.MSELoss(ratings_pred, rating)
                loss.backward()
                optimizer.step()
                scheduler.step()
                client_loss.append(loss.item())