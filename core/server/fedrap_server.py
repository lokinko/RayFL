import random

import torch
import numpy as np
import pandas as pd

from dataset.dataset_factory import DatasetFactory
from core.server.base_server import BaseServer
from core.model.model.build_model import build_model

fedrap_args = {
    "model": "cf",
    "num_users": 6040,
    "num_items": 3706,

    "min_items": 10,
    "num_negatives": 4,
    'item_hidden_dim': 32,
    'negatives_candidates': 99,
}

class FedRapServer(BaseServer):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.args = fedrap_args | args
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.model = build_model(self.args)
        self.dataset = DatasetFactory(self.args).load_dataset()

    def split_data(self):
        data = self.dataset.load_user_dataset(self.args['min_items'])
        if self.args['dataset'] in ['movielens-1m', 'movielens-100k']:
            train_ratings, val_ratings, test_ratings, negatives = data

            self.train_data = {}
            grouped_train_ratings = train_ratings.groupby('userId')
            for userId, user_train_ratings in grouped_train_ratings:
                self.train_data[userId]['userId'] = userId
                self.train_data[userId]['train'] = self._negative_sample(
                    user_train_ratings, self.negatives, self.args['num_negatives'])

            self.val_data = self._negative_sample(val_ratings, negatives, self.args['negatives_candidates'])
            self.test_data = self._negative_sample(test_ratings, negatives, self.args['negatives_candidates'])
        else:
            raise NotImplementedError(f"Dataset {self.args['dataset']} for {self.args['method']} not implemented")

    def _negative_sample(self, pos_ratings: pd.DataFrame, negatives: dict, num_negatives):
        rating_df = pd.merge(pos_ratings, negatives[['userId', 'negative_samples']], on='userId')
        users, items, ratings = [], [], []
        for row in rating_df.itertuples():
            users.append(int(row.userId))
            items.append(int(row.itemId))
            ratings.append(float(row.rating))

            for _, neg_item in enumerate(random.sample(list(row.negative_samples), num_negatives)):
                users.append(int(row.userId))
                items.append(int(neg_item))
                ratings.append(float(0))
        return [users, items, ratings]

    def select_participants(self, client_sample_ratio: float):
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
