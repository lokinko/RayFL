import logging
import importlib
import traceback
from dotenv import load_dotenv
load_dotenv()

import torch

from algorithm import *
from utils.args import get_args
from utils.utils import initLogging, seed_anything

if __name__ == "__main__":
    import ray
    ray.init(num_gpus=torch.cuda.device_count())

    args, _ = get_args()
    seed_anything(seed=args['seed'])
    initLogging(args['log_dir'] / "main.log")

    logging.info(f"{'='*15} Task Configuration Below {'='*15}")
    for arg in args:
        try:
            logging.info(f"{arg:<25}: {str(args[arg]):<25}")
        except Exception:
            logging.error(f"Record task configuration failed, {traceback.format_exc()}")

    algorithm = importlib.import_module(f"algorithm.{args['method']}")
    algorithm.run(args)