import copy
from typing import Tuple

import numpy as np
from core.server.base_server import BaseServer
from core.model import CollaborativeFilterModel

from dataset import DatasetFactory

class FedRapServer(BaseServer):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.global_model = CollaborativeFilterModel(args)
        self.data = DatasetFactory(args).load_dataset()
        self.avaliable_clients = None

    def set_clients_data(self) -> None:
        for user in self.data:
            train_data, test_data, train_sample_size, test_sample_size = \
                self.split_data(self.dataset[user]['user_data'])

            self.clients[user] = {
                'user_id': self.dataset[user]['user_id'],
                'train_data': train_data,
                'train_sample_size': train_sample_size,
                'test_data': test_data,
                'test_sample_size': test_sample_size,
                'model': copy.deepcopy(self.global_model)
            }

    def split_data(self, data) -> Tuple[list, list, int, int]:
        return [], [], 0, 0

    def select_clients(self):
        '''
        Randomly select clients for this round.
        '''
        self.avaliable_clients = np.random.choice(
            list(self.clients.keys()),
            int(len(self.clients) * self.args['clients_sample_ratio']),
            replace=False
        )

    def aggregate(self):
        pass

    def train_one_round(self):
        pass

    def test(self):
        pass