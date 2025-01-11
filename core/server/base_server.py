from abc import ABC, abstractmethod

base_args = {}
'''
method specific arguments should be provieded with dict format.
'''


class BaseServer(ABC):
    def __init__(self, args) -> None:
        self.args = args
        self.train_data = None
        self.val_data = None
        self.test_data = None

        self.global_model = None
        self.user_context = {}
        self.ray_actor_pool = None
        self.metrics = None

    @abstractmethod
    def allocate_init_status(self):
        '''
        Allocate initial status for the server.

        This method should be implemented in the subclass and is responsible for initializing various attributes of the server.
        
        Attributes:
            - self.train_data: [dict] Training data for each participant.
            - self.test_data: [list] Validation data.
            - self.model: [torch.nn.Module] Public model initiated for all participants.
            - self.users: [dict] Personal model for each participant.
            - self.pool: [ray.util.ActorPool] Pool for all actors.
            - self.metrics: [GlobalMetrics] Metrics for evaluation.
        '''

    @abstractmethod
    def select_participants(self):
        pass

    @abstractmethod
    def aggregate(self, participants):
        pass

    @abstractmethod
    def train_on_round(self, participants):
        pass

    @abstractmethod
    def test_on_round(self, model_params, data):
        pass
