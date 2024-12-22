from abc import ABC, abstractmethod

class BaseServer(ABC):
    def __init__(self, args) -> None:
        self.args = args
        self.clients = {}       # client_id -> client_index
        self.train_data = {}    # client_index -> train_data
        self.test_data = {}     # client_index -> test_data
        self.model = {}         # client_index -> client_model

    def select_clients(self):
        pass

    @abstractmethod
    def set_clients_data(self):
        pass

    @abstractmethod
    def aggregate(self):
        pass

    @abstractmethod
    def train_one_round(self):
        pass

    @abstractmethod
    def test(self):
        pass