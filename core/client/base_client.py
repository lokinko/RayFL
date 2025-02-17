from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseClient(ABC):
    def __init__(self, args) -> None:
        self.args = args

    @abstractmethod
    def train(self, model, user_data, **kwargs):
        pass

    def sync_args(self, args: Dict[str, Any]):
        self.args.update(args)