from dataset.movielens import MovieLens

class DatasetFactory(ABC):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.dataset = args['dataset']

    def load_dataset_dict(self):
        if self.dataset == 'movielens':
            dataset = MovieLens(self.args).load_dataset_dict()
        else:
            raise NotImplementedError(f"Dataset {self.dataset} not implemented")
        return dataset