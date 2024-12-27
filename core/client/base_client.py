from abc import ABC, abstractmethod

from utils.logs import initLogging

class BaseClient(ABC):
    def __init__(self, args) -> None:
        self.args = args
        self._name = "base_client"
        initLogging(args['log_dir'] / f"{self._name}_{args['timestamp']}.log")

    @abstractmethod
    def train(self, model, user_data, **kwargs):
        pass