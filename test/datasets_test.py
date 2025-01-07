import unittest

from dataset import MovieLens

dataset_dir = '../data/'

class DatasetsTest(unittest.TestCase):
    def test_movielens(self):
        args = {
            'dataset': 'movielens-1m',
            'min_items': 10,
            'num_negatives': 4,
            'negatives_candidates': 99,
        }

        dataset = MovieLens(args)
        dataset.load_user_dataset(args['min_items'], dataset_dir + f"{args['dataset']}/ratings.dat")
        if args['dataset'] == 'movielens-1m':
            args['num_users'] = len(dataset.user_pool)
            args['num_items'] = len(dataset.item_pool)
            self.assertEqual(args['num_users'], 6040)
            self.assertEqual(args['num_items'], 3706)

        train, val, test = dataset.sample_data()
        for user, data in train.items():
            self.assertEqual(len(data['train']), 3) # user, item, rating
            self.assertEqual(len(data['train'][0]) % (args['num_negatives'] + 1), 0)
            self.assertEqual(len(val[user]['positive_items']), 1)
            self.assertEqual(len(val[user]['negative_items']), args['negatives_candidates'])
            self.assertEqual(len(test[user]['positive_items']), 1)
            self.assertEqual(len(test[user]['negative_items']), args['negatives_candidates'])

if __name__ == "__main__":
    unittest.main()