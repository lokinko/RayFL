
import logging

from core.server.server_factory import ServerFactory

from utils.args import get_args
from utils.logs import initLogging
from utils.seeds import seed_anything

if __name__ == "__main__":
    args, _ = get_args()
    seed_anything(seed=args['seed'])
    initLogging(args['log_dir'] / f"server_{args['timestamp']}.log")

    server_factory = ServerFactory(args)
    server = server_factory.create_server()
    logging.info(f"Server Factory creates {args['method']} server successfully.")

    # for i in range(args['num_rounds']):
    #     server.select_clients()
    #     server.set_clients_data()
    #     server.train_one_round()
    #     server.aggregate()
    #     server.test()