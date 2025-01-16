import os
import logging
import importlib
import traceback
from dotenv import load_dotenv
load_dotenv()

import ray
import torch
import wandb

import algorithm
from utils.args import get_args
from utils.utils import initLogging, seed_anything

if __name__ == "__main__":
    args = get_args()
    seed_anything(seed=args['seed'])
    initLogging(args['log_dir'] / "main.log")

    wandb_name = f"{args['method']}_{args['dataset']}_{args['timestamp']}"
    if args['verbose']:
        wandb.init(project='pfl-rec', name=wandb_name, config=args, mode='online')
    else:
        wandb.init(project='pfl-rec', name=wandb_name, config=args, mode='disabled')

    ray.init(num_gpus=min(args['num_gpus'], torch.cuda.device_count()), ignore_reinit_error=True)

    algorithm = importlib.import_module(f"algorithm.{args['method']}")
    algorithm.run(args)

    wandb.finish()