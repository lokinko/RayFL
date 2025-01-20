import copy
import logging

import torch
import wandb

from core.server.fedbert_server import FedBertServer

def run(args):
    server = FedBertServer(args)
    server.allocate_init_status()
    logging.info(f"Creates {args['method']} server successfully.")

    for communication_round in range(args['num_rounds']):
        # server.train_data = server.dataset.sample_train_data()

        print(f"Round {communication_round} starts.")
        participants = server.select_participants()
        logging.info(f"Round {communication_round}, participants: {len(participants)}")

        server.train_encoder_on_round(participants)
        round_encoder_loss = sum([sum(server.users[user]['encoder_loss'])/len(server.users[user]['encoder_loss']) for user in server.users]) / len(server.users)
        round_encoder_acc = sum([sum(server.users[user]['encoder_acc'])/len(server.users[user]['encoder_acc']) for user in server.users]) / len(server.users)

        # hr, ndcg = server.test_encoder_on_round(server.test_data)
        server_encoder_params = server.aggregate_encoder(participants)

        for _, user in server.users.items():
            user['encoder_dict'].update(server_encoder_params)

        server.encoder.load_state_dict(server.encoder.state_dict() | server_encoder_params)

        logging.info(f"Round = {communication_round}, Encoder_Loss = {round_encoder_loss}, Encoder_Acc = {round_encoder_acc}")
        wandb.log({'round_encoder_loss': round_encoder_loss,
                    'round_encoder_acc': round_encoder_acc},
                    step=communication_round)

        hr, ndcg = server.test_encoder_on_round(server.test_data)
        wandb.log({'HR': hr,
                    'NDCG': ndcg},
                    step=communication_round)
        logging.info(f"Round = {communication_round}, HR = {hr}, NDCG = {ndcg}")
