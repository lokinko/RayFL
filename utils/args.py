import time
import argparse

from pathlib import Path

import torch

METHOD = ['fedavg', 'fedrap']
DATASET = ['mnist', 'movieLens-1m', 'movielens-100k']

work_dir = Path(__file__).resolve().parents[1]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-method', type=str, choices=METHOD, default="fedrap")
    parser.add_argument('-data', "--dataset", choices=DATASET, type=str, default="movielens-1m")
    parser.add_argument('-num_rounds', type=int, default=100)
    parser.add_argument('-lr', type=float, default=0.01)
    parser.add_argument('-cr', '--clients_sample_ratio', type=float, default=1.0)
    parser.add_argument('-bs', '--batch_size', type=int, default=1024)
    parser.add_argument('--local_epoch', type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args, unknown_args = parser.parse_known_args()

    args = vars(args)

    # set the running timestamp
    args['timestamp'] = time.strftime('%m%d%H%M%S', time.localtime(time.time()))

    # set the working directory
    args['work_dir'] = work_dir

    # set the log directory
    args['log_dir'] = Path(
        f"{args['work_dir']}/logs/{args['method'].lower()}_{args['dataset'].lower()}_{args['timestamp']}")

    if not args['log_dir'].exists():
        args['log_dir'].mkdir(parents=True, exist_ok=True)

    return args, unknown_args