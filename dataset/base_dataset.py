from abc import ABC, abstractmethod

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