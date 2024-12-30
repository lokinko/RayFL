import math
import logging

from core.server.fedrap_server import FedRapServer

def run(args):
    server = FedRapServer(args)
    logging.info(f"Creates {args['method']} server successfully.")

    for communication_round in range(args['num_rounds']):
        print(f"Round {communication_round} starts.")
        server.allocate_init_status()
        participants = server.select_participants()
        logging.info(f"Round {communication_round}, participants: {participants}")

        server.train_on_round(participants)
        logging.info(f"Round {communication_round}, train finished.")

        server_params = server.aggregate(participants)
        server.model.load_state_dict(server.model.state_dict() | server_params)
        logging.info(f"Round {communication_round}, aggregate finished.")

        hr, ndcg = server.test(server.test_data)
        logging.info(f"Round {communication_round}, HR: {hr}, NDCG: {ndcg}")

        server.args['lr_network'] = server.args['lr_network'] * server.args['decay_rate']
        server.args['lr_args'] = server.args['lr_args'] * server.args['decay_rate']

        if server.args['vary_param'] == 'tanh':
            server.args['lambda'] = math.tanh(communication_round / 10) * server.args['lambda']
            server.args['mu'] = math.tanh(communication_round / 10) * server.args['mu']
        elif server.args['vary_param'] == 'sin':
            server.args['lambda'] = (math.sin(communication_round / 10) + 1) / 2 * server.args['lambda']
            server.args['mu'] = (math.sin(communication_round / 10) + 1) / 2 * server.args['mu']
        elif server.args['vary_param'] == 'square':
            if communication_round % 5 == 0:
                server.args['lambda'] = 0 if server.args['lambda'] == server.args['lambda'] else server.args['lambda']
                server.args['mu'] = 0 if server.args['mu'] == server.args['mu'] else server.args['mu']
        elif server.args['vary_param'] == 'frac':
            server.args['lambda'] = 1 / (communication_round + 1) * server.args['lambda']
            server.args['mu'] = 1 / (communication_round + 1) * server.args['mu']

if __name__ == '__main__':
    run()