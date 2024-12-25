from dataset import MovieLens

class DatasetFactory:
    def __init__(self, args) -> None:
        self.args = args
        self.dataset = args['dataset']

    def load_dataset(self):
        if self.dataset in ['movielens-1m', 'movielens-100k']:
            dataset = MovieLens(self.args).load_user_dataset()
        else:
            raise NotImplementedError(f"Dataset {self.dataset} not implemented")
        return dataset