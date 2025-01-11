import random

import ray
import numpy
import pandas as pd

from dataset.base_dataset import BaseDataset
from dataset.rec_utils import datasetFilter, reindex, binarize, negative_sample, group_seperate_items_by_ratings
from utils.utils import measure_time, seed_anything, initLogging

@ray.remote
class MovieLens(BaseDataset):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.ratings = None
        self.user_pool = None
        self.item_pool = None
        seed_anything(args['seed'])
        initLogging(args['log_dir'] / "dataset.log", stream=False)

    def load_user_dataset(self, min_items, data_file):
        # origin data with all [uid, mid, rating, timestamp] samples.
        ratings = pd.read_csv(
            data_file, sep='\t', header=None, usecols=[0, 1, 2], names=['uid', 'mid', 'rating'], engine='python')

        rank = ratings[['mid']].drop_duplicates().reindex()
        rank['timestamp'] = np.arange((len(rank)))
        ratings = pd.merge(ratings, rank, on=['mid'], how='left')

        # filter the user with num_samples < min_items
        ratings = datasetFilter(ratings, min_items=min_items)
        self.ratings = reindex(ratings)

        # binarize the ratings, positive click = 1
        preprocess_ratings = binarize(self.ratings)

        # statistic user and item interact
        self.user_pool = set(self.ratings['userId'].unique())
        self.item_pool = set(self.ratings['itemId'].unique())

        self.train_ratings, self.val_ratings, self.test_ratings = self._split_loo(preprocess_ratings)
        self.negatives = self.samples_negative_candidates(self.ratings, self.args['negatives_candidates'])

    def samples_negative_candidates(self, ratings, negatives_candidates: int):
        interact_status = ratings.groupby('userId')['itemId'].apply(set).reset_index().rename(
            columns={'itemId': 'interacted_items'})

        interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: self.item_pool - x)
        interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(list(x), negatives_candidates))
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

    @measure_time()
    def sample_train_data(self):
        grouped_ratings = self.train_ratings.groupby('userId')
        train = {}
        for user_id, user_ratings in grouped_ratings:
            train[user_id] = {}
            train[user_id]['train'] = negative_sample(
                user_ratings, self.negatives[['userId', 'negative_items']], self.args['num_negatives'])
        return train

    @measure_time()
    def sample_test_data(self):
        val_users, val_items, val_ratings = negative_sample(
            self.val_ratings, self.negatives[['userId', 'negative_samples']], self.args['negatives_candidates'], mode="test")
        val = group_seperate_items_by_ratings(val_users, val_items, val_ratings)

        test_users, test_items, test_ratings = negative_sample(
            self.test_ratings, self.negatives[['userId', 'negative_samples']], self.args['negatives_candidates'], mode="test")
        test = group_seperate_items_by_ratings(test_users, test_items, test_ratings)
        return val, test