import random

import ray
import pandas as pd

from utils.utils import *
from utils.rec_utils import *

@ray.remote
class UserItemRatingsDataset:
    def __init__(self, args) -> None:
        self.args = args
        self.ratings = None
        self.user_pool = None
        self.item_pool = None
        seed_anything(args['seed'])
        initLogging(args['log_dir'] / f"{args['dataset']}.log", stream=False)

    def load_user_dataset(self, min_items, data_file):
        # origin data with all [uid, mid, rating, timestamp] samples.
        if self.args['dataset'] == 'movielens-100k':
            ratings = pd.read_csv(
                data_file, sep=',', header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')

        elif self.args['dataset'] == 'movielens-1m':
            ratings = pd.read_csv(
                data_file, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')

        elif self.args['dataset'] == 'amazon':
            ratings = pd.read_csv(
                data_file, sep=',', header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')

        elif self.args['dataset'] == 'last.fm':
            ratings = pd.read_csv(
                data_file, sep="\t", header=1, usecols=[0, 1, 2], names=['uid', 'mid', 'rating'], engine='python')

            # take the item orders instead of real timestamp
            rank = ratings[['mid']].drop_duplicates().reindex()
            rank['timestamp'] = np.arange((len(rank)))
            ratings = pd.merge(ratings, rank, on=['mid'], how='left')

        elif self.args['dataset'] == 'tenrec':
            chunks = pd.read_csv(data_file, sep=",", header=1, usecols=[0, 1, 2],
                                names=['uid', 'mid', 'rating'], engine='python', chunksize=1000000)

            all_chunks = []
            for chunk in chunks:
                all_chunks.append(chunk)

            ratings = pd.concat(all_chunks)

            # take the item orders instead of real timestamp
            rank = ratings[['mid']].drop_duplicates().reindex()
            rank['timestamp'] = np.arange((len(rank)))
            ratings = pd.merge(ratings, rank, on=['mid'], how='left')

        else:
            raise ValueError(f"Invalid dataset: {self.args['dataset']}")

        # filter the user with num_samples < min_items
        ratings = datasetFilter(ratings, min_items=min_items)
        self.ratings = reindex(ratings)

        # binarize the ratings, positive click = 1
        preprocess_ratings = binarize(self.ratings)

        # statistic user and item interact
        self.user_pool = set(self.ratings['userId'].unique())
        self.item_pool = set(self.ratings['itemId'].unique())

        self.train_ratings, self.val_ratings, self.test_ratings = self._split_loo(preprocess_ratings, self.args['boardline'])

        self.train_negatives = self.samples_negative_candidates(self.train_ratings, self.args['negatives_candidates'])
        # self.train_negatives = self.samples_negative_candidates(self.ratings, self.args['negatives_candidates'])
        self.test_negatives = self.samples_negative_candidates(self.train_ratings, self.args['negatives_candidates'])
        return len(self.user_pool), len(self.item_pool)

    def samples_negative_candidates(self, ratings, negatives_candidates: int):
        interact_status = ratings.groupby('userId')['itemId'].apply(set).reset_index().rename(
            columns={'itemId': 'interacted_items'})

        interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: self.item_pool - x)
        interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(list(x), negatives_candidates))
        return interact_status[['userId', 'negative_items', 'negative_samples']]

    def _split_loo(self, ratings, boardline=2):
        ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
        test = ratings[ratings['rank_latest'] < boardline]
        val = ratings[ratings['rank_latest'] == boardline]
        train = ratings[ratings['rank_latest'] > boardline]

        assert train['userId'].nunique() == test['userId'].nunique() == val['userId'].nunique()
        assert len(train) + len(test) + len(val) == len(ratings)

        return train[['userId', 'itemId', 'rating']], val[['userId', 'itemId', 'rating']], test[
            ['userId', 'itemId', 'rating']]


    @measure_time()
    def sample_federated_train_data(self):
        grouped_ratings = self.train_ratings.groupby('userId')
        train = {}
        for user_id, user_ratings in grouped_ratings:
            rating_df = pd.merge(user_ratings, self.train_negatives[['userId', 'negative_items']], on='userId')
            users, items, ratings = [], [], []
            for row in rating_df.itertuples():
                users.append(int(row.userId))
                items.append(int(row.itemId))
                ratings.append(float(row.rating))

                negative_samples = random.sample(list(row.negative_items), self.args['num_negatives'])

                for _, neg_item in enumerate(negative_samples):
                    users.append(int(row.userId))
                    items.append(int(neg_item))
                    ratings.append(float(0))

            train[user_id] = (users, items, ratings)
        return train


    @measure_time()
    def sample_central_train_data(self):
        rating_df = pd.merge(self.train_ratings, self.train_negatives[['userId', 'negative_items']], on='userId')
        users, items, ratings = [], [], []
        for row in rating_df.itertuples():
            users.append(int(row.userId))
            items.append(int(row.itemId))
            ratings.append(float(row.rating))

            negative_samples = random.sample(list(row.negative_items), self.args['num_negatives'])

            for _, neg_item in enumerate(negative_samples):
                users.append(int(row.userId))
                items.append(int(neg_item))
                ratings.append(float(0))
        return (users, items, ratings)


    @measure_time()
    def sample_val_data(self):
        rating_df = pd.merge(self.val_ratings, self.test_negatives[['userId', 'negative_samples']], on='userId')
        users, items, ratings = [], [], []
        for row in rating_df.itertuples():
            users.append(int(row.userId))
            items.append(int(row.itemId))
            ratings.append(float(row.rating))

            negative_samples = row.negative_samples

            for _, neg_item in enumerate(negative_samples):
                users.append(int(row.userId))
                items.append(int(neg_item))
                ratings.append(float(0))

        val = group_seperate_items_by_ratings(users, items, ratings)
        return val


    @measure_time()
    def sample_test_data(self):
        rating_df = pd.merge(self.test_ratings, self.test_negatives[['userId', 'negative_samples']], on='userId')
        users, items, ratings = [], [], []
        for row in rating_df.itertuples():
            users.append(int(row.userId))
            items.append(int(row.itemId))
            ratings.append(float(row.rating))

            negative_samples = row.negative_samples

            for _, neg_item in enumerate(negative_samples):
                users.append(int(row.userId))
                items.append(int(neg_item))
                ratings.append(float(0))
        test = group_seperate_items_by_ratings(users, items, ratings)
        return test
