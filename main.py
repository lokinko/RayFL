import importlib
from dotenv import load_dotenv
load_dotenv()

from algorithm import *
from utils.args import get_args
from utils.init import init_all

if __name__ == "__main__":
    import ray
    ray.init(num_gpus=1)

    args, _ = get_args()
    init_all(args, args['log_dir'] / "main.log")

    algorithm = importlib.import_module(f"algorithm.{args['method']}")
    algorithm.run(args)