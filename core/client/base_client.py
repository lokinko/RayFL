from abc import ABC, abstractmethod

class BaseClient(ABC):
    def __init__(self, args, model) -> None:
        self.args = args
        self.model = model

    @abstractmethod
    def train(self, train_data, **kwargs):
        pass

    @abstractmethod
    def test(self, test_data, **kwargs):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass