from abc import ABC, abstractmethod

class BaseServer(ABC):
    def __init__(self, args) -> None:
        self.args = args
        self.model = None            # global model should be initialized in subclass

    @abstractmethod
    def select_participants(self):
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