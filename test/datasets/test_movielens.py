import unittest

from pathlib import Path

import ray

from dataset.movielens import MovieLens

dataset_dir = '../../dataset/data/'

class MovielensTest(unittest.TestCase):
    def setUp(self):
        ray.init(ignore_reinit_error=True)
        self.args = {
            'seed': 0,
            'num_negatives': 4,
            'negatives_candidates': 99,
            'log_dir': Path('../../logs/'),
        }

    def test_movielens100k(self):
        self.args['dataset'] = 'movielens-100k'
        dataset = MovieLens.remote(self.args)
        ray.get(dataset.load_user_dataset.remote(min_items=10, data_file=dataset_dir + "movielens-100k/ratings.dat"))
        self.assertEqual(len(ray.get(dataset.get_user_pool.remote())), 943)
        self.assertEqual(len(ray.get(dataset.get_item_pool.remote())), 1682)

        train = ray.get(dataset.sample_federated_train_data.remote())
        val, test = ray.get(dataset.sample_test_data.remote())
        for user, data in train.items():
            self.assertEqual(len(data['train']), 3) # user, item, rating
            self.assertEqual(len(data['train'][0]) % (self.args['num_negatives'] + 1), 0)
            self.assertEqual(len(val[user]['positive_items']), 1)
            self.assertEqual(len(val[user]['negative_items']), self.args['negatives_candidates'])
            self.assertEqual(len(test[user]['positive_items']), 1)
            self.assertEqual(len(test[user]['negative_items']), self.args['negatives_candidates'])

    def test_movielens1m(self):
        self.args['dataset'] = 'movielens-1m'
        dataset = MovieLens.remote(self.args)
        ray.get(dataset.load_user_dataset.remote(min_items=10, data_file=dataset_dir + "movielens-1m/ratings.dat"))
        self.assertEqual(len(ray.get(dataset.get_user_pool.remote())), 6040)
        self.assertEqual(len(ray.get(dataset.get_item_pool.remote())), 3706)

        train = ray.get(dataset.sample_federated_train_data.remote())
        val, test = ray.get(dataset.sample_test_data.remote())
        for user, data in train.items():
            self.assertEqual(len(data['train']), 3) # user, item, rating
            self.assertEqual(len(data['train'][0]) % (self.args['num_negatives'] + 1), 0)
            self.assertEqual(len(val[user]['positive_items']), 1)
            self.assertEqual(len(val[user]['negative_items']), self.args['negatives_candidates'])
            self.assertEqual(len(test[user]['positive_items']), 1)
            self.assertEqual(len(test[user]['negative_items']), self.args['negatives_candidates'])

if __name__ == "__main__":
    unittest.main()