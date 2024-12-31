import time
import random
import logging

import torch
import numpy as np

def measure_time(repeats: int=1):
    def decorator(func):
        def wrapper(*args, **kwargs):
            timings = []
            for _ in range(repeats):
                start_time = time.time()
                func_res = func(*args, **kwargs)
                end_time = time.time()
                elapsed = end_time - start_time
                timings.append(elapsed)
            np_times = np.array(timings)
            average_time = np.mean(np_times)
            variance = np.var(np_times)
            logging.info(f"[{func.__name__}] Average time over {repeats} runs: {average_time:.6f} seconds")
            if repeats > 1:
                logging.info(f"[{func.__name__}] Variance of times: {variance:.6f}")
            return func_res
        return wrapper
    return decorator


def seed_anything(seed):
    # set global random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def initLogging(file_name, level=logging.INFO, storage=True, stream=True):
    logger = logging.getLogger()
    logger.setLevel(level)

    # Create stream handlers
    if stream:
        s_handler = logging.StreamHandler()
        s_format = logging.Formatter('[%(asctime)s] %(levelname)s | %(message)s')
        s_handler.setFormatter(s_format)
        logger.addHandler(s_handler)

    # Create file_handler
    if storage:
        f_handler = logging.FileHandler(file_name)
        f_format = logging.Formatter('[%(asctime)s] %(levelname)s | %(message)s')
        f_handler.setFormatter(f_format)
        logger.addHandler(f_handler)