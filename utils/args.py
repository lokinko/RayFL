import time
import argparse

from pathlib import Path

METHOD = ['fedavg', 'fedrap']
DATASET = ['MNIST', 'MovieLens-1m']

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('--model', type=str, default='CNN')
    parser.add_argument('-method', type=str, choices=METHOD, default="fedavg")
    parser.add_argument('-data', "--dataset", type=str, default="MNIST")
    parser.add_argument('-num_rounds', type=int, default=100)
    parser.add_argument('-num_clients', type=int, default=100)

    parser.add_argument('-lr', type=float, default=0.01)

    # For recommendation system
    parser.add_argument('-num_items', type=int, default=16)
    parser.add_argument('-item_hidden_dim', type=int, default=32)

    args, unknown_args = parser.parse_known_args()

    args = vars(args)
    args['timestamp'] = time.strftime('%m%d%H%M%S', time.localtime(time.time()))
    args['log_dir'] = Path(f"logs/{args['method'].lower()}_{args['model'].lower()}_{args['dataset'].lower()}_{args['timestamp']}")
    if not args['log_dir'].exists():
        args['log_dir'].mkdir(parents=True, exist_ok=True)
    return args, unknown_args