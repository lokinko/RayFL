
from core.server.base_server import BaseServer
from core.model import CollaborativeFilterModel

class FedRapServer(BaseServer):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.global_model = CollaborativeFilterModel(args)

    def select_clients(self):
        pass

    def set_clients_data(self):
        pass

    def aggregate(self):
        pass

    def train_one_round(self):
        pass

    def test(self):
        pass