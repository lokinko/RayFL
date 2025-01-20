import copy
import random

import numpy as np
import pandas as pd
from dataset.base_dataset import BaseDataset

class Lastfm(BaseDataset):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.ratings = None
        self.user_pool = None
        self.item_pool = None

    def load_user_dataset(self, min_items, data_file):
        # origin data with all [uid, mid, rating, timestamp] samples.
        data = pd.read_csv(
            data_file, sep=',', header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')

        # filter the user with num_samples < min_items
        ratings = self.datasetFilter(data, min_items=min_items)
        self.ratings = self.reindex(ratings)

        # binarize the ratings, positive click = 1
        preprocess_ratings = self._binarize(self.ratings)

        # statistic user and item interact
        self.user_pool = set(self.ratings['userId'].unique())
        self.item_pool = set(self.ratings['itemId'].unique())

        num_users = len(self.user_pool)
        num_items = len(self.item_pool)
        num_interactions = len(ratings)

        print(f"Number of users: {num_users}")
        print(f"Number of items: {num_items}")
        print(f"Number of interactions: {num_interactions}")

        # create negative item samples for model learning
        # 99 negatives for each user's test item
        self.negatives = self._sample_negative_candidates(self.ratings, self.args['negatives_candidates'])

        self.train_ratings, self.val_ratings, self.test_ratings = self._split_loo(preprocess_ratings)

        return None

    def reindex(self, ratings):
        # Reindex user id and item id
        user_id = ratings[['uid']].drop_duplicates().reindex()
        user_id['userId'] = np.arange(len(user_id))
        ratings = pd.merge(ratings, user_id, on=['uid'], how='left')

        item_id = ratings[['mid']].drop_duplicates()
        item_id['itemId'] = np.arange(1, len(item_id)+1)
        ratings = pd.merge(ratings, item_id, on=['mid'], how='left')

        # ratings = ratings[['userId', 'itemId', 'rating', 'timestamp']].sort_values(by='userId', ascending=True)
        ratings = ratings[['userId', 'itemId', 'rating', 'timestamp']].sort_values(by=['userId', 'timestamp'], ascending=True)
        return ratings

    def sample_train_data(self):
        grouped_ratings = self.train_ratings.groupby('userId')
        train = {}
        for user_id, user_ratings in grouped_ratings:
            train[user_id] = {}
            train[user_id]['train'] = self._negative_sample(
                user_ratings, self.negatives, self.args['num_negatives'])

            # unique_item_ids = list(dict.fromkeys(user_ratings['itemId']))
            # train[user_id]['train_positive'] = unique_item_ids
            train[user_id]['train_positive'] = user_ratings.itemId.tolist()

        return train

    