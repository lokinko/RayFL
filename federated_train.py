import logging
import importlib
import traceback

from dotenv import load_dotenv
load_dotenv()

import ray
import torch
import wandb

from utils.args import get_args
from utils.utils import initLogging, seed_anything

if __name__ == "__main__":
    args = get_args()
    seed_anything(seed=args['seed'])
    initLogging(args['log_dir'] / "main.log")

    records = f"{args['method']}_{args['timestamp']}"
    wandb.init(project=f"Dataset = [{args['dataset']}]", name=records, mode='online')

    ray.init(num_gpus = min(args['num_gpus'], torch.cuda.device_count()), ignore_reinit_error=True)

    try:
        algorithm = importlib.import_module(f"core.algorithm.{args['method']}")
        algorithm.run(args)
    except Exception:
        logging.error(f"Run algorithm {args['method']} failed, {traceback.format_exc()}")
    finally:
        wandb.finish()
