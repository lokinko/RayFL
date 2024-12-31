import importlib
from dotenv import load_dotenv
load_dotenv()

from algorithm import *
from utils.args import get_args
from utils.utils import initLogging, seed_anything

if __name__ == "__main__":
    import ray
    ray.init(num_gpus=1)

    args, _ = get_args()
    seed_anything(seed=args['seed'])
    initLogging(args['log_dir'] / "main.log")

    algorithm = importlib.import_module(f"algorithm.{args['method']}")
    algorithm.run(args)