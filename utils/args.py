import os
import time
import argparse

from pathlib import Path

import torch

METHOD = ['fedncf', 'fedrap', 'fedaug', 'fedbert']
DATASET = ['movielens-1m', 'movielens-100k', 'lastfm-2k', 'amazon', 'foursquare']

work_dir = Path(__file__).resolve().parents[1]

# if work_dir.as_posix() not in os.environ['PYTHONPATH']:
#     os.environ['PYTHONPATH'] = f"{os.environ['PYTHONPATH']}:{work_dir.as_posix()}"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-method', type=str, choices=METHOD, default="fedbert")
    parser.add_argument('-data', "--dataset", choices=DATASET, type=str, default="movielens-100k")
    parser.add_argument('-min_items', type=int, default=10)
    parser.add_argument('-num_rounds', type=int, default=100)
    parser.add_argument('-lr', type=float, default=0.01)
    parser.add_argument('-cr', '--client_sample_ratio', type=float, default=1.0)
    parser.add_argument('-bs', '--batch_size', type=int, default=2048)
    parser.add_argument('--local_epoch', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--num_gpus', type=int, default=torch.cuda.device_count())
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--item_hidden_dim', type=int, default=32)
    
    # BERT
    parser.add_argument('--bert_max_len', type=int, default=100, help='Length of sequence for bert')
    parser.add_argument('--num_items', type=int, default=1682, help='Number of total items')
    parser.add_argument('--bert_hidden_units', type=int, default=32, help='Size of hidden vectors (d_model)')
    parser.add_argument('--bert_num_blocks', type=int, default=1, help='Number of transformer layers')
    parser.add_argument('--bert_num_heads', type=int, default=1, help='Number of heads for multi-attention')
    parser.add_argument('--bert_dropout', type=float, default=0.1, help='Dropout probability to use throughout the model')
    parser.add_argument('--bert_mask_prob', type=float, default=0.15, help='Probability for masking items in the training sequence')
    parser.add_argument('--per_head', type=bool, default=False)
    args, unknown_args = parser.parse_known_args()

    args = vars(args)
    # set special 
    if args['dataset'] == 'movielens-1m':
        args['num_users'] = 6040
        args['num_items'] = 3706
    elif args['dataset'] == 'movielens-100k':
        args['num_users'] = 943
        args['num_items'] = 1682
    elif args['dataset'] == 'lastfm-2k':
        args['num_users'] = 1600
        args['num_items'] = 12454
    elif args['dataset'] == 'amazon':
        args['num_users'] = 8072
        args['num_items'] = 11830
    elif args['dataset'] == 'foursquare':
        args['num_users'] = 1083
        args['num_items'] = 38333
        
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