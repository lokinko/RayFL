import ray

from core.client.base_client import BaseClient

@ray.remote
class FedRapActor(BaseClient):
    def __init__(self, args, model, **kwargs) -> None:
        super().__init__(args, model)
        

    def train(self, train_data, **kwargs):
        pass

    def test(self, test_data, **kwargs):
        pass