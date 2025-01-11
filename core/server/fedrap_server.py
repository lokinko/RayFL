import copy

import ray
import torch
import numpy as np

from tqdm import tqdm

from core.server.base_server import BaseServer
from core.client.fedrap_client import FedRapActor
from core.model.model.build_model import build_model
from dataset import MovieLens
from utils.metrics.metronatk import GlobalMetrics
from utils.utils import seed_anything, initLogging, measure_time

special_args = {
    'model': 'cf',
    'num_negatives': 4,
    'item_hidden_dim': 32,
    'negatives_candidates': 99,

    'top_k': 10,
    'regular': 'l1',
    'lr_network': 0.5,
    'lr_args': 1,
    'l2_regularization': 1e-4,
    'lambda': 0.01,
    'mu': 0.01,
    'vary_param': 'tanh',
    'decay_rate': 0.97,
    'tol': 0.0001,
}

class FedRapServer(BaseServer):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.args = special_args | self.args
        seed_anything(seed=self.args['seed'])
        initLogging(args['log_dir'] / "server.log", stream=False)

    def allocate_init_status(self):
        self.dataset = self.load_dataset()
        self.train_data = ray.get(self.dataset.sample_train_data.remote())
        self.val_data, self.test_data = ray.get(self.dataset.sample_test_data.remote())
        self.model = build_model(self.args)

        for user in self.train_data:
            self.users[user] = {
                'user_id': user,
                'model_dict': copy.deepcopy(self.model.state_dict())}

        self.pool = ray.util.ActorPool([
            FedRapActor.options(
                num_cpus=0.2, num_gpus=self.args['num_gpus'] / float(self.args['num_workers'])).remote(self.args)
                for _ in range(self.args['num_workers'])])
        self.metrics = GlobalMetrics(self.args['top_k'])

    @measure_time()
    def load_dataset(self):
        if self.args['dataset'] == 'movielens-1m':
            self.args['num_users'], self.args['num_items'], self.args['min_items'] = 6040, 3706, 10

            dataset = MovieLens.remote(self.args)
            ray.get(dataset.load_user_dataset.remote(
                self.args['min_items'], self.args['work_dir']/'data/movielens-1m/ratings.dat'))

        elif self.args['dataset'] == 'movielens-100k':
            dataset = MovieLens.remote(self.args)
            ray.get(dataset.load_user_dataset(
                self.args['min_items'], self.args['work_dir']/'data/movielens-100k/ratings.dat'))

        else:
            raise NotImplementedError(f"Dataset {self.args['dataset']} for {self.args['method']} not implemented")
        return dataset


    def select_participants(self):
        participants = np.random.choice(
            list(self.users), int(len(self.users) * self.args['client_sample_ratio']), replace=False)
        return participants


    @torch.no_grad()
    def aggregate(self, participants, keys):
        assert participants is not None, "No participants selected for aggregation."

        samples, global_weights = 0, {}

        for user in tqdm(participants, desc="Aggregating", ncols=120):
            for key in keys:
                if key not in global_weights:
                    global_weights[key] = torch.zeros_like(self.model.state_dict()[key])

                global_weights[key] += self.users[user]['model_dict'][key] * len(self.train_data[user]['train'])
            samples += len(self.train_data[user]['train'])

        global_weights = {k: v / samples for k, v in global_weights.items()}
        return global_weights


    def train_on_round(self, participants):
        results = self.pool.map_unordered(
            lambda a, v: a.train.remote(copy.deepcopy(self.model), v), \
            [(self.users[user_id], self.train_data[user_id]) for user_id in participants])

        results = tqdm(results, total=len(participants), ncols=120)
        for result in results:
            user_id, client_model, client_loss = result
            self.users[user_id]['model_dict'] = copy.deepcopy(client_model.state_dict())
            self.users[user_id]['loss'] = client_loss
            results.set_description(f"Training loss: {client_loss[-1]:.4f}")

    @torch.no_grad()
    def test_on_round(self, user_ratings: dict):
        test_scores = None
        negative_scores = None
        test_users, test_items, negative_users, negative_items = None, None, None, None

        for user, user_data in tqdm(user_ratings.items(), desc="Testing", ncols=120):
            # load each user's mlp parameters.
            user_model = copy.deepcopy(self.model)

            user_param_dict = self.users[user]['model_dict']
            user_param_dict['item_commonality.weight'] = copy.deepcopy(self.model.state_dict()['item_commonality.weight'])
            user_model.load_state_dict(user_param_dict)

            user_model.eval()

            test_score, _, _ = user_model(user_data['positive_items'])
            negative_score, _, _ = user_model(user_data['negative_items'])

            if test_scores is None:
                test_scores = test_score
                negative_scores = negative_score
                test_users = torch.tensor([user] * len(test_score))
                negative_users = torch.tensor([user] * len(negative_score))
                test_items = torch.tensor(user_data['positive_items'])
                negative_items = torch.tensor(user_data['negative_items'])
            else:
                test_scores = torch.cat((test_scores, test_score))
                negative_scores = torch.cat((negative_scores, negative_score))
                test_users = torch.cat((test_users, torch.tensor([user] * len(test_score))))
                negative_users = torch.cat((negative_users, torch.tensor([user] * len(negative_score))))
                test_items = torch.cat((test_items, torch.tensor(user_data['positive_items'])))
                negative_items = torch.cat((negative_items, torch.tensor(user_data['negative_items'])))

        self.metrics.subjects = [test_users.data.view(-1).tolist(),
                                 test_items.data.view(-1).tolist(),
                                 test_scores.data.view(-1).tolist(),
                                 negative_users.data.view(-1).tolist(),
                                 negative_items.data.view(-1).tolist(),
                                 negative_scores.data.view(-1).tolist()]

        hr, ndcg = self.metrics.cal_hit_ratio(), self.metrics.cal_ndcg()

        return hr, ndcg

    @torch.no_grad()
    def test_commonality(self, user_ratings: dict):
        test_scores = None
        negative_scores = None
        test_users, test_items, negative_users, negative_items = None, None, None, None

        for user, user_data in tqdm(user_ratings.items(), desc="commonality testing", ncols=120):
            item_commonality = copy.deepcopy(self.model.item_commonality)

            # load each user's mlp parameters.
            user_model = copy.deepcopy(self.model)

            user_param_dict = self.users[user]['model_dict']
            user_model.load_state_dict(user_param_dict)
            user_model.setItemCommonality(item_commonality)

            user_model.eval()

            test_score, _, = user_model.commonality_forward(user_data['positive_items'])
            negative_score, _, = user_model.commonality_forward(user_data['negative_items'])

            if test_scores is None:
                test_scores = test_score
                negative_scores = negative_score
                test_users = torch.tensor([user] * len(test_score))
                negative_users = torch.tensor([user] * len(negative_score))
                test_items = torch.tensor(user_data['positive_items'])
                negative_items = torch.tensor(user_data['negative_items'])
            else:
                test_scores = torch.cat((test_scores, test_score))
                negative_scores = torch.cat((negative_scores, negative_score))
                test_users = torch.cat((test_users, torch.tensor([user] * len(test_score))))
                negative_users = torch.cat((negative_users, torch.tensor([user] * len(negative_score))))
                test_items = torch.cat((test_items, torch.tensor(user_data['positive_items'])))
                negative_items = torch.cat((negative_items, torch.tensor(user_data['negative_items'])))

        self.metrics.subjects = [test_users.data.view(-1).tolist(),
                                 test_items.data.view(-1).tolist(),
                                 test_scores.data.view(-1).tolist(),
                                 negative_users.data.view(-1).tolist(),
                                 negative_items.data.view(-1).tolist(),
                                 negative_scores.data.view(-1).tolist()]

        hr, ndcg = self.metrics.cal_hit_ratio(), self.metrics.cal_ndcg()

        return hr, ndcg


    @torch.no_grad()
    def test_decouple(self, participants, user_ratings: dict):
        test_scores = None
        negative_scores = None
        test_users, test_items, negative_users, negative_items = None, None, None, None

        avg_item_personal = self.aggregate(participants, keys=['item_personality.weight'])

        for user, user_data in tqdm(user_ratings.items(), desc="commonality testing", ncols=120):
            # load each user's mlp parameters.
            user_model = copy.deepcopy(self.model)

            user_param_dict = copy.deepcopy(self.users[user]['model_dict'])
            user_param_dict['item_commonality.weight'] = copy.deepcopy(self.model.state_dict()['item_commonality.weight']) \
                                                        + avg_item_personal['item_personality.weight']
            user_model.load_state_dict(user_param_dict)

            user_model.eval()

            test_score, _, = user_model.commonality_forward(user_data['positive_items'])
            negative_score, _, = user_model.commonality_forward(user_data['negative_items'])

            if test_scores is None:
                test_scores = test_score
                negative_scores = negative_score
                test_users = torch.tensor([user] * len(test_score))
                negative_users = torch.tensor([user] * len(negative_score))
                test_items = torch.tensor(user_data['positive_items'])
                negative_items = torch.tensor(user_data['negative_items'])
            else:
                test_scores = torch.cat((test_scores, test_score))
                negative_scores = torch.cat((negative_scores, negative_score))
                test_users = torch.cat((test_users, torch.tensor([user] * len(test_score))))
                negative_users = torch.cat((negative_users, torch.tensor([user] * len(negative_score))))
                test_items = torch.cat((test_items, torch.tensor(user_data['positive_items'])))
                negative_items = torch.cat((negative_items, torch.tensor(user_data['negative_items'])))

        self.metrics.subjects = [test_users.data.view(-1).tolist(),
                                 test_items.data.view(-1).tolist(),
                                 test_scores.data.view(-1).tolist(),
                                 negative_users.data.view(-1).tolist(),
                                 negative_items.data.view(-1).tolist(),
                                 negative_scores.data.view(-1).tolist()]

        hr, ndcg = self.metrics.cal_hit_ratio(), self.metrics.cal_ndcg()

        return hr, ndcg, avg_item_personal
