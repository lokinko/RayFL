import logging


from core.server.fedrap_server import FedRapServer
from utils.args import get_args
from utils.init import init_all

def run():
    args, _ = get_args()
    init_all(args, args['log_dir'] / "main.log")

    server = FedRapServer(args)
    server.split_data()
    logging.info(f"Server Factory creates {args['method']} server successfully.")

    for i in range(args['num_rounds']):
        participants = server.select_participants()
        logging.info(f"Round {i}, participants: {participants}")

        server.train_one_round(participants)
        logging.info(f"Round {i}, train finished.")

        server.aggregate(participants)
        logging.info(f"Round {i}, aggregate finished.")

        hr, ndcg = server.test(server.data['test'])
        logging.info(f"Round {i}, HR: {hr}, NDCG: {ndcg}")