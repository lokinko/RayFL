import logging
from abc import ABC, abstractmethod

from core.server.base_server import BaseServer
from core.server.fedrap_server import FedRapServer

from utils.logs import initLogging

class ServerFactory(ABC):
    def __init__(self, args):
        self.args = args
        self.method = args['method']
        # initLogging(args['log_dir'] / f"server_{args['timestamp']}.log")

    def create_server(self) -> BaseServer:
        if self.method == 'fedrap':
            server = FedRapServer(self.args)
        else:
            raise NotImplementedError(f"Method {self.method} not implemented")
        return server

    def create_model(self):
        pass

    def create_data_loader(self):
        pass