import copy
import random
import logging

import ray
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm

from core.server.base_server import BaseServer
from core.client.fedbert_client import FedBertActor
from core.model.model.build_model import build_model
from dataset import MovieLens, Lastfm, Amazon, Foursquare
from utils.metrics.metronatk import GlobalMetrics
from utils.utils import seed_anything, initLogging, measure_time

special_args = {
    'model': 'pcf',
    'num_users': 943,
    'num_items': 1682,
    'min_items': 10,
    'num_negatives': 4,
    'item_hidden_dim': 32,
    'negatives_candidates': 99,

    'top_k': 10,
    'regular': 'l1',
    'lr_network': 0.1,
    'lr_args': 0.1,
    'l2_regularization': 1e-4,
    'lambda': 0.1,
    'mu': 0.1,
    'vary_param': 'tanh',
    'decay_rate': 0.97,
    'tol': 0.0001,
}

class FedBertServer(BaseServer):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.args = special_args | self.args
        seed_anything(seed=self.args['seed'])
        initLogging(args['log_dir'] / "server.log", stream=False)

    def allocate_init_status(self):
        self.dataset = self.load_dataset()
        # self.train_data, self.val_data, self.test_data = self.dataset.sample_data()
        self.train_data = self.dataset.sample_train_data()
        self.test_data = self.dataset.test_data
        self.encoder, self.model = build_model(self.args)

        for user in self.train_data:
            self.users[user] = {
                'user_id': user,
                'encoder_dict': copy.deepcopy(self.encoder.state_dict()),
                'model_dict': copy.deepcopy(self.model.state_dict())}

        self.pool = ray.util.ActorPool([FedBertActor.remote(self.args) for _ in range(self.args['num_workers'])])
        self.metrics = GlobalMetrics(self.args['top_k'])

        self.max_len = 100
        self.mask_token = self.args['num_items']+1
        self.per_head = self.args['per_head']

    @measure_time()
    def load_dataset(self):
        if self.args['dataset'] == 'movielens-1m':
            dataset = MovieLens(self.args)
            dataset.load_user_dataset(self.args['min_items'], self.args['work_dir']/'data/movielens-1m/ratings.dat')

        elif self.args['dataset'] == 'movielens-100k':
            dataset = MovieLens(self.args)
            dataset.load_user_dataset(self.args['min_items'], self.args['work_dir']/'data/movielens-100k/ratings.data')
        
        elif self.args['dataset'] == 'lastfm-2k':
            dataset = Lastfm(self.args)
            dataset.load_user_dataset(self.args['min_items'], self.args['work_dir']/'data/lastfm-2k/ratings.dat')

        elif self.args['dataset'] == 'amazon':
            dataset = Amazon(self.args)
            dataset.load_user_dataset(self.args['min_items'], self.args['work_dir']/'data/amazon/ratings.dat')
        
        elif self.args['dataset'] == 'foursquare':
            dataset = Foursquare(self.args)
            dataset.load_user_dataset(self.args['min_items'], self.args['work_dir']/'data/foursquare/ratings.dat')

        else:
            raise NotImplementedError(f"Dataset {self.args['dataset']} for {self.args['method']} not implemented")
        return dataset


    def select_participants(self):
        participants = np.random.choice(
            list(self.users), int(len(self.users) * self.args['client_sample_ratio']), replace=False)
        return participants

    def aggregate_encoder(self, participants):
        assert participants is not None, "No participants selected for aggregation."

        global_encoder_state = {}
        if self.per_head:
            excluded_param = ['out.weight', 'out.bias']
        else:
            excluded_param = [None]

        for param_name, param in self.encoder.state_dict().items():
            if param_name not in excluded_param:
                global_encoder_state[param_name] = torch.zeros_like(param)

        samples = 0
        for user in tqdm(participants, desc="Aggregating"):
            user_samples = len(self.train_data[user]['train_positive'])
            for param_name in global_encoder_state.keys():
                global_encoder_state[param_name] += self.users[user]['encoder_dict'][param_name] * user_samples
            samples += user_samples

        # Average the accumulated gradients by the total number of samples
        for param_name in global_encoder_state.keys():
            global_encoder_state[param_name] /= samples

        return global_encoder_state


    def train_encoder_on_round(self, participants):
        results = self.pool.map_unordered(
            lambda a, v: a.train_encoder.remote(copy.deepcopy(self.encoder), v), \
            [(self.users[user_id], self.train_data[user_id]) for user_id in participants])
        for result in tqdm(results, desc="Training", total=len(participants)):
            user_id, client_encoder, client_encoder_loss, client_encoder_acc = result
            # user_id, client_encoder, client_encoder_loss = result
            self.users[user_id]['encoder_dict'].update(client_encoder.state_dict())
            self.users[user_id]['encoder_loss'] = client_encoder_loss
            self.users[user_id]['encoder_acc'] = client_encoder_acc
    
    @torch.no_grad()
    def test_encoder_on_round(self, user_ratings: dict):
        test_scores = None
        negative_scores = None
        test_users, test_items, negative_users, negative_items = None, None, None, None

        for user, user_data in tqdm(user_ratings.items(), desc="Testing"):
            # load each user's mlp parameters.
            user_encoder = copy.deepcopy(self.encoder)

            user_param_dict = self.users[user]['encoder_dict']
            # user_param_dict['item_commonality.weight'] = self.model.state_dict()['item_commonality.weight']
            user_encoder.load_state_dict(user_param_dict)

            user_encoder.eval()

            seq = self.train_data[user]['train_positive']
            seq = seq + [self.mask_token]
            seq = seq[-self.max_len:]
            padding_len = self.max_len - len(seq)
            seq = [0] * padding_len + seq

            scores = user_encoder(torch.tensor(seq).unsqueeze(0))  # B x T x V [1, 100, 1683]
            scores = scores[:, -1, :]  # B x V [1, 1683]
            # scores = scores.gather(1, candidates)  # B x C

            test_score = scores[0][user_data['positive_items']]
            negative_score = scores[0][user_data['negative_items']]

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
    def test_on_round(self, user_ratings: dict):
        test_scores = None
        negative_scores = None
        test_users, test_items, negative_users, negative_items = None, None, None, None

        for user, user_data in tqdm(user_ratings.items(), desc="Testing"):
            # load each user's mlp parameters.
            user_model = copy.deepcopy(self.model)

            user_param_dict = self.users[user]['model_dict']
            user_param_dict['item_commonality.weight'] = self.model.state_dict()['item_commonality.weight']
            user_model.load_state_dict(user_param_dict)

            user_model.eval()

            test_score, _ = user_model(user_data['positive_items'])
            negative_score, _ = user_model(user_data['negative_items'])

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
    


    def train_on_round(self, participants):
        results = self.pool.map_unordered(
            lambda a, v: a.train.remote(copy.deepcopy(self.model), v), \
            [(self.users[user_id], self.train_data[user_id]) for user_id in participants])
        for result in tqdm(results, desc="Training", total=len(participants)):
            user_id, client_model, client_loss = result
            self.users[user_id]['model_dict'].update(client_model.state_dict())
            self.users[user_id]['loss'] = client_loss
    
    @torch.no_grad()
    def aggregate(self, participants):
        assert participants is not None, "No participants selected for aggregation."

        samples = 0
        global_item_community_weight = torch.zeros_like(self.model.item_commonality.weight)
        for user in tqdm(participants, desc="Aggregating"):
            global_item_community_weight += self.users[user]['model_dict']['item_commonality.weight'] \
                                            * len(self.train_data[user]['train'])
            samples += len(self.train_data[user]['train'])
        global_item_community_weight /= samples
        return {'item_commonality.weight': global_item_community_weight}