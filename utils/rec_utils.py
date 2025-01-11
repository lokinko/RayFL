import copy
import random

import numpy as np
import pandas as pd



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
