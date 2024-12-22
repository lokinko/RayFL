from dataset.base_dataset import BaseDataset

class MovieLens(BaseDataset):
    def __init__(self, args) -> None:
        super().__init__(args)

    def load_dataset_list(self):
        '''
        movielens-1m: https://files.grouplens.org/datasets/movielens/ml-1m.zip
        movielens-100k: https://files.grouplens.org/datasets/movielens/ml-100k.zip
        load the rating.data file, return a dict with keys: user_id, item_id, rating, timestamp.
        '''
        return None