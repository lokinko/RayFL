import logging

from utils.logs import initLogging
from utils.seeds import seed_anything

def init_all(args, log_file):
    initLogging(log_file)
    seed_anything(seed=args['seed'])