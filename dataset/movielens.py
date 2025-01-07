import copy
import random

import ray
import numpy as np
import pandas as pd

from dataset.base_dataset import BaseDataset
from utils.utils import measure_time, seed_anything, initLogging

def datasetFilter(ratings, min_items=5):
    # filter unuseful data
    ratings = ratings[ratings['rating'] > 0]

    # only keep users who rated at least {self.min_items} items
    user_count = ratings.groupby('uid').size()
    user_subset = np.in1d(ratings.uid, user_count[user_count >= min_items].index)
    filter_ratings = ratings[user_subset].reset_index(drop=True)

    return filter_ratings


def reindex(ratings):
    # Reindex user id and item id
    user_id = ratings[['uid']].drop_duplicates().reindex()
    user_id['userId'] = np.arange(len(user_id))
    ratings = pd.merge(ratings, user_id, on=['uid'], how='left')

    item_id = ratings[['mid']].drop_duplicates()
    item_id['itemId'] = np.arange(len(item_id))
    ratings = pd.merge(ratings, item_id, on=['mid'], how='left')

    ratings = ratings[['userId', 'itemId', 'rating', 'timestamp']].sort_values(by='userId', ascending=True)
    return ratings


def binarize(ratings):
    data = copy.deepcopy(ratings)
    data.loc[data['rating'] > 0, 'rating'] = 1.0
    return data


def negative_sample(pos_ratings: pd.DataFrame, negatives: pd.DataFrame, num_negatives, mode="train"):
    rating_df = pd.merge(pos_ratings, negatives, on='userId')
    users, items, ratings = [], [], []
    for row in rating_df.itertuples():
        users.append(int(row.userId))
        items.append(int(row.itemId))
        ratings.append(float(row.rating))

        if mode == "train":
            negative_samples = random.sample(list(row.negative_items), num_negatives)
        elif mode == "test":
            negative_samples = row.negative_samples
        else:
            raise ValueError("mode should be 'train' or 'test'")

        for _, neg_item in enumerate(negative_samples):
            users.append(int(row.userId))
            items.append(int(neg_item))
            ratings.append(float(0))
    return (users, items, ratings)


def group_seperate_items_by_ratings(users, items, ratings):
    user_dict = {}
    for (user, item, rating) in zip(users, items, ratings):
        if user not in user_dict:
            user_dict[user] = {'positive_items': [], 'negative_items': []}
        if rating == 1:
            user_dict[user]['positive_items'].append(item)
        else:
            user_dict[user]['negative_items'].append(item)
    return user_dict


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
        data = pd.read_csv(
            data_file, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')

        # filter the user with num_samples < min_items
        ratings = datasetFilter(data, min_items=min_items)
        self.ratings = reindex(ratings)

        # binarize the ratings, positive click = 1
        preprocess_ratings = binarize(self.ratings)

        # statistic user and item interact
        self.user_pool = set(self.ratings['userId'].unique())
        self.item_pool = set(self.ratings['itemId'].unique())

        self.train_ratings, self.val_ratings, self.test_ratings = self._split_loo(preprocess_ratings)
        self.test_negative = self.samples_negative_candidates(self.ratings, self.args['negatives_candidates'])

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
                user_ratings, self.test_negative[['userId', 'negative_items']], self.args['num_negatives'])
        return train

    @measure_time()
    def sample_test_data(self):
        val_users, val_items, val_ratings = negative_sample(
            self.val_ratings, self.test_negative[['userId', 'negative_samples']], self.args['negatives_candidates'], mode="test")
        val = group_seperate_items_by_ratings(val_users, val_items, val_ratings)

        test_users, test_items, test_ratings = negative_sample(
            self.test_ratings, self.test_negative[['userId', 'negative_samples']], self.args['negatives_candidates'], mode="test")
        test = group_seperate_items_by_ratings(test_users, test_items, test_ratings)
        return val, test