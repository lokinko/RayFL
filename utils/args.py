import os
import time
import argparse

from datetime import datetime
from pathlib import Path

import torch

METHOD = ['fedrap', 'pfedrec', 'fedpor', 'fedpora']
DATASET = ['movielens-100k', 'movielens-1m', 'amazon', 'last.fm', 'tenrec']

work_dir = Path(__file__).resolve().parents[1]

if work_dir.as_posix() not in os.environ.get('PYTHONPATH', {}):
    os.environ['PYTHONPATH'] = f"{os.environ['PYTHONPATH']}:{work_dir.as_posix()}"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-lr', type=float, default=0.01)
    parser.add_argument('-cr', '--client_sample_ratio', type=float, default=1.0)
    parser.add_argument('-bs', '--batch_size', type=int, default=2048)

    parser.add_argument('--method', type=str, choices=METHOD)
    parser.add_argument('--dataset', type=str, choices=DATASET)
    parser.add_argument('--data_file', type=str)
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--num_rounds', type=int, default=100)
    parser.add_argument('--local_epoch', type=int, default=10)

    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--num_gpus', type=int, default=torch.cuda.device_count())

    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--comment', type=str, default='default')

    args, unknown_args = parser.parse_known_args()

    args = vars(args)

    # set the working directory
    args['work_dir'] = work_dir
    args['log_dir'] = work_dir / 'logs'
    args['data_dir'] = work_dir / 'dataset/data'

    # set the log directory
    args['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(datetime.now().timestamp() + 8 * 3600))
    args['log_dir'] = Path(f"{args['log_dir']}/{args['method']}/{args['dataset']}_{args['timestamp']}")

    if not args['log_dir'].exists():
        args['log_dir'].mkdir(parents=True, exist_ok=True)

    if unknown_args is not None:
        args['unknown'] = {}
        for k, v in zip(unknown_args[::2], unknown_args[1::2]):
            args['unknown'][k[2:]] = v
    return args