import copy
import logging

import torch

from core.server.fedrap_server import FedRapServer

def run(args):
    server = FedRapServer(args)
    server.allocate_init_status()
    logging.info(f"Creates {args['method']} server successfully.")

    for communication_round in range(args['num_rounds']):
        print(f"Round {communication_round} starts.")
        participants = server.select_participants()
        logging.info(f"Round {communication_round}, participants: {participants}")

        server.train_on_round(participants)
        round_loss = sum([sum(server.users[user]['loss'])/len(server.users[user]['loss']) for user in server.users]) / len(server.users)

        origin_params = copy.deepcopy(server.model.state_dict())
        server_params = server.aggregate(participants)

        for _, user in server.users.items():
            user['model_dict'].update(server_params)

        server.model.load_state_dict(server.model.state_dict() | server_params)

        hr, ndcg = server.test_on_round(server.test_data)
        logging.info(f"Round = {communication_round}, Loss = {round_loss}, HR = {hr}, NDCG = {ndcg}")

        server.args['lr_network'] = server.args['lr_network'] * server.args['decay_rate']
        server.args['lr_args'] = server.args['lr_args'] * server.args['decay_rate']
        server.train_data, server.val_data, server.test_data = server.dataset.sample_data()

        # if server.args['vary_param'] == 'tanh':
        #     server.args['lambda'] = math.tanh(communication_round / 10) * server.args['lambda']
        #     server.args['mu'] = math.tanh(communication_round / 10) * server.args['mu']
        # elif server.args['vary_param'] == 'sin':
        #     server.args['lambda'] = (math.sin(communication_round / 10) + 1) / 2 * server.args['lambda']
        #     server.args['mu'] = (math.sin(communication_round / 10) + 1) / 2 * server.args['mu']
        # elif server.args['vary_param'] == 'square':
        #     if communication_round % 5 == 0:
        #         server.args['lambda'] = 0 if server.args['lambda'] == server.args['lambda'] else server.args['lambda']
        #         server.args['mu'] = 0 if server.args['mu'] == server.args['mu'] else server.args['mu']
        # elif server.args['vary_param'] == 'frac':
        #     server.args['lambda'] = 1 / (communication_round + 1) * server.args['lambda']
        #     server.args['mu'] = 1 / (communication_round + 1) * server.args['mu']

        save_path = server.args['log_dir'] / f"{communication_round}" / f"{communication_round}.pt"

        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "origin_params": origin_params,
                "aggregate_params": server_params,
                "updated_params": server.model.state_dict(),
                "participants": participants,
                "users": server.users,
                "args": server.args,
                "data": [server.train_data, server.val_data, server.test_data],
                "metrics": [hr, ndcg]
            }, save_path
        )