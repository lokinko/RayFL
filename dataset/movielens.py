import copy
import random

import numpy as np
import pandas as pd
from dataset.base_dataset import BaseDataset

class MovieLens(BaseDataset):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.origin_data = None
        self.user_pool = None
        self.item_pool = None

    def load_user_dataset(self, min_items, data_file):
        # origin data with all [uid, mid, rating, timestamp] samples.
        self.origin_data = pd.read_csv(
            data_file, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')

        # filter the user with num_samples < min_items
        ratings = self.datasetFilter(self.origin_data, min_items=min_items)
        ratings = self.reindex(ratings)

        # binarize the ratings, positive click = 1
        preprocess_ratings = self._binarize(ratings)

        # statistic user and item interact
        self.user_pool = set(ratings['userId'].unique())
        self.item_pool = set(ratings['itemId'].unique())

        negatives_candidates = self._samples_negative(ratings, self.args['negatives_candidates'])

        train_ratings, val_ratings, test_ratings = self._split_loo(preprocess_ratings)
        return train_ratings, val_ratings, test_ratings, negatives_candidates

    def datasetFilter(self, ratings, min_items=5):
        # filter unuseful data
        ratings = ratings[ratings['rating'] > 0]

        # only keep users who rated at least {self.min_items} items
        user_count = ratings.groupby('uid').size()
        user_subset = np.in1d(ratings.uid, user_count[user_count >= min_items].index)
        filter_ratings = ratings[user_subset].reset_index(drop=True)

        return filter_ratings

    def reindex(self, ratings):
        # Reindex user id and item id
        user_id = ratings[['uid']].drop_duplicates().reindex()
        user_id['userId'] = np.arange(len(user_id))
        ratings = pd.merge(ratings, user_id, on=['uid'], how='left')

        item_id = ratings[['mid']].drop_duplicates()
        item_id['itemId'] = np.arange(len(item_id))
        ratings = pd.merge(ratings, item_id, on=['mid'], how='left')

        ratings = ratings[['userId', 'itemId', 'rating', 'timestamp']].sort_values(by='userId', ascending=True)
        return ratings

    def _binarize(self, ratings):
        data = copy.deepcopy(ratings)
        data.loc[data['rating'] > 0, 'rating'] = 1.0
        return data

    def _samples_negative(self, ratings, negatives_candidates: int):
        interact_status = ratings.groupby('userId')['itemId'].apply(set).reset_index().rename(
            columns={'itemId': 'interacted_items'})

        interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: self.item_pool - x)
        interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(x, negatives_candidates))
        return interact_status[['userId', 'negative_items', 'negative_samples']]

    def _split_loo(self, ratings):
        ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
        test = ratings[ratings['rank_latest'] == 1]
        val = ratings[ratings['rank_latest'] == 2]
        train = ratings[ratings['rank_latest'] > 2]

        assert train['userId'].nunique() == test['userId'].nunique() == val['userId'].nunique()
        assert len(train) + len(test) + len(val) == len(ratings)

        return train[['userId', 'itemId', 'rating']], val[['userId', 'itemId', 'rating']], test[
            ['userId', 'itemId', 'rating']]