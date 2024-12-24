from abc import ABC, abstractmethod
from collections import defaultdict

class BaseServer(ABC):
    def __init__(self, args) -> None:
        self.args = args
        self.clients = defaultdict(dict)    # client_id -> client_index
        self.global_model = None            # global model should be initialized in subclass

    @abstractmethod
    def set_clients_data(self):
        '''
        Set train_data and test_data for each client, should be implemented in subclass.
        Must set self.clients with the map of client_id -> client_index'''

    @abstractmethod
    def split_data(self, data, **kwargs):
        '''
        Split data into train_data and test_data, should be implemented in subclass.
        '''

    @abstractmethod
    def select_clients(self):
        '''
        Select clients for this round, should be implemented in subclass.
        '''

    @abstractmethod
    def aggregate(self):
        pass

    @abstractmethod
    def train_one_round(self):
        pass

    @abstractmethod
    def test(self):
        pass