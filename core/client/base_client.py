from abc import ABC, abstractmethod

class BaseClient(ABC):
    def __init__(self, args) -> None:
        self.args = args

    @abstractmethod
    def train(self, model, user_data, **kwargs):
        pass