import numpy as np
import pandas as pd
from dataset.base_dataset import BaseDataset

class MovieLens(BaseDataset):
    def __init__(self, args) -> None:
        super().__init__(args)

    def load_user_dataset(self):
        '''
        movielens-1m: https://files.grouplens.org/datasets/movielens/ml-1m.zip
        movielens-100k: https://files.grouplens.org/datasets/movielens/ml-100k.zip
        load the rating.data file, return a dict with keys: user_id, item_id, rating, timestamp.
        '''
        if self.dataset == 'movielens-1m':
            data = self.args['work_dir'] / 'data/movielens-1m/ratings.dat'
        elif self.dataset == 'movielens-100k':
            data = self.args['work_dir'] / 'data/movielens-1m/ratings.dat'
        else:
            raise NotImplementedError(f"Movielens dataset has no {self.dataset} implemented")
        ratings = pd.read_csv(data, sep='::', header=None, names=['user', 'item', 'rating', 'timestamp'], engine='python')

        # Reindex user id and item id
        user_id = ratings[['user']].drop_duplicates().reindex()
        user_id['userId'] = np.arange(len(user_id))
        ratings = pd.merge(ratings, user_id, on=['user'], how='left')

        item_id = ratings[['item']].drop_duplicates()
        item_id['itemId'] = np.arange(len(item_id))
        ratings = pd.merge(ratings, item_id, on=['item'], how='left')
        ratings = ratings[['userId', 'itemId', 'rating', 'timestamp']].sort_values(by='userId', ascending=True)

        user_ratings = {
            user_id: {
                'user_id': str(df['user'].values[0]),
                'user_data': df[['itemId', 'rating', 'timestamp']].sort_values(by='timestamp')
            }
            for user_id, df in ratings.groupby('userId')
        }

        return user_ratings