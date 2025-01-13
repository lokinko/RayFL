from abc import ABC, abstractmethod

import copy
import random

import numpy as np
import pandas as pd

class BaseDataset(ABC):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.dataset = args['dataset']

    @abstractmethod
    def load_user_dataset(self):
        '''
        Abstract method to load dataset, should be implemented in subclass
        dataset should be returned with the following format:
        {
            '1': {
                'user_id': 'real_client_id',
                'user_data': user's local dataset
            }, ...
        }
        '''

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

        # ratings = ratings[['userId', 'itemId', 'rating', 'timestamp']].sort_values(by='userId', ascending=True)
        ratings = ratings[['userId', 'itemId', 'rating', 'timestamp']].sort_values(by=['userId', 'timestamp'], ascending=True)
        return ratings

    def _binarize(self, ratings):
        data = copy.deepcopy(ratings)
        data.loc[data['rating'] > 0, 'rating'] = 1.0
        return data
    
    def _split_loo(self, ratings):
        ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
        test = ratings[ratings['rank_latest'] == 1]
        val = ratings[ratings['rank_latest'] == 2]
        train = ratings[ratings['rank_latest'] > 2]

        assert train['userId'].nunique() == test['userId'].nunique() == val['userId'].nunique()
        assert len(train) + len(test) + len(val) == len(ratings)

        return train[['userId', 'itemId', 'rating']], val[['userId', 'itemId', 'rating']], test[
            ['userId', 'itemId', 'rating']]

    def _sample_negative_candidates(self, ratings, negatives_candidates: int):
        interact_status = ratings.groupby('userId')['itemId'].apply(set).reset_index().rename(
            columns={'itemId': 'interacted_items'})

        interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: self.item_pool - x)
        interact_status['negative_candidates'] = interact_status['negative_items'].apply(lambda x: random.sample(x, negatives_candidates))
        return interact_status[['userId', 'negative_items', 'negative_candidates']]

    def _negative_sample(self, pos_ratings: pd.DataFrame, negatives: dict, num_negatives):
        rating_df = pd.merge(pos_ratings, negatives[['userId', 'negative_items']], on='userId')
        users, items, ratings = [], [], []
        cnt = 0
        uid =-1
        for row in rating_df.itertuples():
            users.append(int(row.userId))
            items.append(int(row.itemId))
            uid = row.userId
            
            ratings.append(float(row.rating))
            if float(row.rating) != 1.0: 
                # print(row.userId, row.itemId, row.rating)
                cnt = cnt + 1
                continue
            for _, neg_item in enumerate(random.sample(list(row.negative_items), num_negatives)):
                users.append(int(row.userId))
                items.append(int(neg_item))
                ratings.append(float(0))
        # if cnt !=0:
        #     print(uid)
        #     print(cnt)
        return (users, items, ratings)
    
    def sample_train_data(self):
        grouped_ratings = self.train_ratings.groupby('userId')
        train = {}
        for user_id, user_ratings in grouped_ratings:
            train[user_id] = {}
            train[user_id]['train'] = self._negative_sample(
                user_ratings, self.negatives, self.args['num_negatives'])
            train[user_id]['train_positive'] = user_ratings.itemId.tolist()

        return train

    @property
    def test_data(self):
        rating_df = pd.merge(self.test_ratings, self.negatives[['userId', 'negative_candidates']], on='userId')
        users, items, ratings = [], [], []
        for row in rating_df.itertuples():
            users.append(int(row.userId))
            items.append(int(row.itemId))
            ratings.append(float(row.rating))

            for _, neg_item in enumerate(row.negative_candidates):
                users.append(int(row.userId))
                items.append(int(neg_item))
                ratings.append(float(0))
        test = self.group_seperate_items_by_ratings(users, items, ratings)
        return test

    def group_seperate_items_by_ratings(self, users, items, ratings):
        user_dict = {}
        for (user, item, rating) in zip(users, items, ratings):
            if user not in user_dict:
                user_dict[user] = {'positive_items': [], 'negative_items': []}
            if rating == 1:
                user_dict[user]['positive_items'].append(item)
            else:
                user_dict[user]['negative_items'].append(item)
        return user_dict