import copy

import ray
import torch
import numpy as np

from tqdm import tqdm

from core.server.base_server import BaseServer
from core.client.fedpor_client import FedPORActor
from dataset import MovieLens, AmazonVideo
from model.recommendation import PersonalRegularUserItemInteraction
from utils.metrics.metronatk import GlobalMetrics
from utils.utils import seed_anything, initLogging, measure_time

class FedPORServer(BaseServer):
    def __init__(self, args, special_args) -> None:
        super().__init__(args, special_args)
        seed_anything(seed=self.args['seed'])
        initLogging(args['log_dir'] / "server.log", stream=False)

    def allocate_init_status(self):
        self.dataset = self.load_dataset()
        self.train_data = ray.get(self.dataset.sample_train_data.remote())
        self.val_data, self.test_data = ray.get(self.dataset.sample_test_data.remote())
        self.global_model = PersonalRegularUserItemInteraction(self.args)

        for user_id in range(int(self.args['num_users'])):
            self.user_context[user_id] = {
                'user_id': user_id,
                'state_dict': copy.deepcopy(self.global_model.state_dict()),
                'loss': [],
            }

        actor_cpus, actor_gpus = 0.5, self.args['num_gpus'] / float(self.args['num_workers'])
        self.ray_actor_pool = ray.util.ActorPool([
            FedPORActor.options(num_cpus=actor_cpus, num_gpus=actor_gpus).remote(self.args)
            for _ in range(self.args['num_workers'])])
        self.metrics = GlobalMetrics(self.args['top_k'])


    @measure_time()
    def load_dataset(self):
        if self.args['dataset'] == 'movielens-1m':
            self.args['min_items'] = 10
            dataset = MovieLens.remote(self.args)
            self.args['num_users'], self.args['num_items'] = ray.get(dataset.load_user_dataset.remote(
                self.args['min_items'], self.args['data_dir'] / 'movielens-1m/ratings.dat'))

        elif self.args['dataset'] == 'movielens-100k':
            self.args['min_items'] = 10
            dataset = MovieLens.remote(self.args)
            self.args['num_users'], self.args['num_items'] = ray.get(dataset.load_user_dataset(
                self.args['min_items'], self.args['data_dir'] / 'movielens-100k/ratings.dat'))

        elif self.args['dataset'] == 'amazon-video':
            self.args["min_items"] = 10
            dataset = AmazonVideo.remote(self.args)
            self.args['num_users'], self.args['num_items'] = ray.get(dataset.load_user_dataset.remote(
                self.args['min_items'], self.args['data_dir'] / 'amazon-video/ratings.csv'))

        else:
            raise NotImplementedError(f"Dataset {self.args['dataset']} for {self.args['method']} not implemented")
        return dataset


    def select_participants(self):
        participants = np.random.choice(
            list(self.user_context), int(len(self.user_context) * self.args['client_sample_ratio']), replace=False)
        return participants


    @torch.no_grad()
    def aggregate(self, participants, keys):
        assert participants is not None, "No participants selected for aggregation."

        samples, global_weights = 0, {}

        iter_participants = tqdm(participants, ncols=120)
        for user in iter_participants:
            for key in keys:
                if key not in global_weights:
                    global_weights[key] = torch.zeros_like(self.global_model.state_dict()[key])

                global_weights[key] += self.user_context[user]['state_dict'][key]
            samples += 1
            iter_participants.set_description(f"Aggregating {samples} participants")

        global_weights = {k: v / samples for k, v in global_weights.items()}
        return global_weights


    def train_on_round(self, participants):
        results = self.ray_actor_pool.map_unordered(
            lambda a, v: a.train.remote(copy.deepcopy(self.global_model), v), \
            [(self.user_context[user_id], self.train_data[user_id]) for user_id in participants])

        results = tqdm(results, total=len(participants), ncols=120)
        for result in results:
            user_id, client_model, client_loss = result
            self.user_context[user_id]['state_dict'] = copy.deepcopy(client_model.state_dict())
            self.user_context[user_id]['loss'] = client_loss
            results.set_description(f"Training loss: {client_loss[-1]:.4f}")


    @torch.no_grad()
    def test_on_round(self, user_ratings: dict):
        test_scores = None
        negative_scores = None
        test_users, test_items, negative_users, negative_items = None, None, None, None

        iter_user_ratings = tqdm(user_ratings.items(), ncols=120)
        for user, user_data in iter_user_ratings:
            # load each user's mlp parameters.
            iter_user_ratings.set_description(f"Testing user {user}")
            user_model = copy.deepcopy(self.global_model)

            user_param_dict = self.user_context[user]['state_dict']
            user_param_dict['item_commonality.weight'] = copy.deepcopy(
                    self.global_model.state_dict()['item_commonality.weight'])
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
        item_commonality = copy.deepcopy(self.global_model.item_commonality)

        iter_user_ratings = tqdm(user_ratings.items(), ncols=120)
        for user, user_data in iter_user_ratings:
            iter_user_ratings.set_description(f"Commonality testing user {user}")

            # load each user's mlp parameters.
            user_model = copy.deepcopy(self.global_model)

            user_state_dict = copy.deepcopy(self.user_context[user]['state_dict'])
            user_model.load_state_dict(user_state_dict)
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
            user_model = copy.deepcopy(self.global_model)

            user_state_dict = copy.deepcopy(self.user_context[user]['state_dict'])
            user_state_dict['item_commonality.weight'] = copy.deepcopy(self.global_model.state_dict()['item_commonality.weight']) \
                                                        + avg_item_personal['item_personality.weight']
            user_model.load_state_dict(user_state_dict)

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
