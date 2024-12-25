import random

import torch
import numpy as np
import pandas as pd

from dataset import DatasetFactory
from core.server.base_server import BaseServer
from core.model import CollaborativeFilterModel

fedrap_args = {
    "min_items": 10,
    "num_negatives": 4,
    'negatives_candidates': 99
}

class FedRapServer(BaseServer):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.args = fedrap_args | args      # add specific args for FedRap
        self.model = CollaborativeFilterModel(args)
        self.dataset = DatasetFactory(args).load_dataset()
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def split_data(self):
        data = self.dataset.load_user_dataset(self.args['min_items'])
        if self.args['dataset'] in ['movielens-1m', 'movielens-100k']:
            train_ratings, val_ratings, test_ratings, negatives = data
            self.train_data = self.get_train_data(train_ratings, negatives, self.args['num_negatives'])
            self.val_data = self.get_val_data(val_ratings, negatives)
            self.test_data = self.get_val_data(test_ratings, negatives)
        else:
            raise NotImplementedError(f"Dataset {self.args['dataset']} for {self.args['method']} not implemented")

    def get_train_data(self, train_ratings: pd.DataFrame, negatives: dict, num_negatives):
        train_ratings = pd.merge(train_ratings, negatives[['userId', 'negative_items']], on='userId')

        train_ratings['negatives'] = train_ratings['negative_items'].apply(lambda x: random.sample(x, num_negatives))
        grouped_train_ratings = train_ratings.groupby('userId')
        train_users = []
        users, items, ratings = [], [], []
        for userId, user_train_ratings in grouped_train_ratings:
            single_user, user_item, user_rating = [], [], []
            train_users.append(userId)
            user_length = len(user_train_ratings)
            for row in user_train_ratings.itertuples():
                single_user.append(int(row.userId))
                user_item.append(int(row.itemId))
                user_rating.append(float(row.rating))
                for i in range(num_negatives):
                    single_user.append(int(row.userId))
                    user_item.append(int(row.negatives[i]))
                    user_rating.append(float(0))  # negative samples get 0 rating
            assert len(single_user) == len(user_item) == len(user_rating)
            assert (1 + num_negatives) * user_length == len(single_user)
            users.append(single_user)
            items.append(user_item)
            ratings.append(user_rating)
        return [users, items, ratings]

    def get_val_data(self, val_ratings: pd.DataFrame, negatives: dict):
        val_ratings = pd.merge(val_ratings, negatives[['userId', 'negative_samples']], on='userId')
        val_users, val_items, negative_users, negative_items = [], [], [], []
        for row in val_ratings.itertuples():
            val_users.append(int(row.userId))
            val_items.append(int(row.itemId))
            for _, neg_item in enumerate(row.negative_samples):
                negative_users.append(int(row.userId))
                negative_items.append(int(neg_item))
        return [torch.LongTensor(val_users), torch.LongTensor(val_items), torch.LongTensor(negative_users),
                torch.LongTensor(negative_items)]

    def select_participants(self, client_sample_ratio: float) -> np.Array:
        '''
        select clients for this round.
        '''
        participants = np.random.choice(
            list(self.clients.keys()), int(len(self.clients) * client_sample_ratio), replace=False)
        return participants

    def aggregate(self):
        pass

    def train_one_round(self):
        pass

    def test(self):
        pass
