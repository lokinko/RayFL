from abc import ABC

from core.server.fedrap_server import FedRapServer

class ServerFactory(ABC):
    def __init__(self, args):
        self.args = args
        self.method = args['method']

    def create_server(self):
        if self.method == 'fedrap':
            server = FedRapServer(self.args)
        else:
            raise NotImplementedError(f"Method {self.method} not implemented")
        assert server.model is not None, f"{self.method} should initialize the model in subclass"
        return server