from abc import ABC, abstractmethod

from core.server.base_server import BaseServer
from core.server.fedrap_server import FedRapServer

class ServerFactory(ABC):
    def __init__(self, args):
        self.args = args
        self.method = args['method']

    def create_server(self) -> BaseServer:
        if self.method == 'fedrap':
            server = FedRapServer(self.args)
        else:
            raise NotImplementedError(f"Method {self.method} not implemented")
        assert server.model is not None, f"{self.method} should initialize the model in subclass"
        return server

    def create_model(self):
        pass

    def create_data_loader(self):
        pass