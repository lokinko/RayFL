import math
import logging

from core.server.fedrap_server import FedRapServer
from utils.args import get_args
from utils.init import init_all

def run():
    args, _ = get_args()
    init_all(args, args['log_dir'] / "main.log")

    server = FedRapServer(args)
    server.allocate_init_status()
    logging.info(f"Server Factory creates {args['method']} server successfully.")

    for round in range(args['num_rounds']):
        print(f"Round {round} starts.")
        participants = server.select_participants()
        logging.info(f"Round {round}, participants: {participants}")

        server.train_on_round(participants)
        logging.info(f"Round {round}, train finished.")

        server.aggregate(participants)
        logging.info(f"Round {round}, aggregate finished.")

        hr, ndcg = server.test(server.test_data)
        logging.info(f"Round {round}, HR: {hr}, NDCG: {ndcg}")

        server.args['lr_network'] = server.args['lr_network'] * server.args['decay_rate']
        server.args['lr_args'] = server.args['lr_args'] * server.args['decay_rate']

        if server.args['vary_param'] == 'tanh':
            server.args['lambda'] = math.tanh(round / 10) * server.args['lambda']
            server.args['mu'] = math.tanh(round / 10) * server.args['mu']
        elif server.args['vary_param'] == 'sin':
            server.args['lambda'] = (math.sin(round / 10) + 1) / 2 * server.args['lambda']
            server.args['mu'] = (math.sin(round / 10) + 1) / 2 * server.args['mu']
        elif server.args['vary_param'] == 'square':
            if round % 5 == 0:
                server.args['lambda'] = 0 if server.args['lambda'] == server.args['lambda'] else server.args['lambda']
                server.args['mu'] = 0 if server.args['mu'] == server.args['mu'] else server.args['mu']
        elif server.args['vary_param'] == 'frac':
            server.args['lambda'] = 1 / (round + 1) * server.args['lambda']
            server.args['mu'] = 1 / (round + 1) * server.args['mu']

if __name__ == '__main__':
    run()