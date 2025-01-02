import copy
import logging

import torch

from core.server.fedaug_server import FedAugServer

def run(args):
    server = FedAugServer(args)
    server.allocate_init_status()
    logging.info(f"Creates {args['method']} server successfully.")

    for communication_round in range(5):
        print(f"Round {communication_round} starts.")
        participants = server.select_participants()
        logging.info(f"Round {communication_round}, participants: {len(participants)}")

        server.train_decoder_on_round(participants)
        round_decoder_loss = sum([sum(server.users[user]['decoder_loss'])/len(server.users[user]['decoder_loss']) for user in server.users]) / len(server.users)
        server_decoder_params = server.aggregate_decoder(participants)
        
        print(server_decoder_params.keys())
        for _, user in server.users.items():
            user['decoder_dict'].update(server_decoder_params)

        server.decoder.load_state_dict(server.decoder.state_dict() | server_decoder_params)

        save_path = server.args['log_dir'] / f"{communication_round}" / f"decoder_{communication_round}.pt"

        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "aggregate_params": server_decoder_params,
                "updated_params": server.decoder.state_dict(),
                "users": server.users,
                "args": server.args,
                "data": [server.train_data, server.val_data, server.test_data],
            }, save_path
        )

    for communication_round in range(args['num_rounds']):
        print(f"Round {communication_round} starts.")
        participants = server.select_participants()
        logging.info(f"Round {communication_round}, participants: {len(participants)}")

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