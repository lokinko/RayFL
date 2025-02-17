import copy
import logging

import ray
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import wandb

from core.server.fedrap_server import FedRAPServer

FEDRAP_ARGS = {
    'num_negatives': 4,
    'item_hidden_dim': 32,
    'negatives_candidates': 99,

    'top_k': 10,
    'tol': 0.0001,

    'mu': 0.01,
    'lambda': 0.01,
    'decay_rate': 0.97,
    'regular': 'l1',
    'vary_param': 'tanh',

    'lr_args': 1e3,
    'lr_network': 0.5,
    'l2_regularization': 1e-4,
}

def run(args):
    server = FedRAPServer(args, FEDRAP_ARGS)
    server.allocate_init_status()

    logging.info(f"Creates {args['method']} server successfully.")

    for communication_round in range(1, args['num_rounds']+1):
        logging.info(f"{'='*20} Round {communication_round} starts {'='*20}")
        participants = server.select_participants()

        # get async train data before training
        train_data_future = server.dataset.sample_federated_train_data.remote()
        server.train_on_round(participants)

        # calculate round loss
        round_loss = 0.0
        for user_id in participants:
            round_loss += sum(server.user_context[user_id]['loss']) / len(server.user_context[user_id]['loss'])
        wandb.log({'round_loss': round_loss / len(participants)}, step=communication_round)

        # aggregate global parameters
        global_params = server.aggregate(participants, ['item_commonality.weight'])

        # update user context
        for _, user_context in server.user_context.items():
            user_context['state_dict']['item_commonality.weight'] = copy.deepcopy(
                                global_params['item_commonality.weight'])

        # update global model
        server_params = copy.deepcopy(server.global_model.state_dict())
        server_params['item_commonality.weight'] = global_params['item_commonality.weight'].data
        server.global_model.load_state_dict(server_params)

        # test on validation and test data
        val_hr, val_ndcg = server.test_on_round(server.val_data)
        test_hr, test_ndcg = server.test_on_round(server.test_data)
        logging.info(f"Val HR = {val_hr:.4f}, Val NDCG = {val_ndcg:.4f}, Test HR = {test_hr:.4f}, Test NDCG = {test_ndcg:.4f}")
        wandb.log({'val_hr': val_hr, 'val_ndcg': val_ndcg, 'test_hr': test_hr, 'test_ndcg': test_ndcg}, step=communication_round)

        # test commonality
        com_hr, com_ndcg = server.test_commonality(server.test_data)
        logging.info(f"Commonality HR = {com_hr:.4f}, Commonality NDCG = {com_ndcg:.4f}")
        wandb.log({'com_hr': com_hr, 'com_ndcg': com_ndcg}, step=communication_round)

        common_figure = plt.figure(figsize=(10, 10))
        sns.heatmap(torch.abs(global_params['item_commonality.weight'][:, :]))

        client_id = 100
        client_personal_fig = plt.figure(figsize=(10, 10))
        sns.heatmap(torch.abs(server.user_context[client_id]['state_dict']['item_personality.weight'][:, :]))

        client_common_fig = plt.figure(figsize=(10, 10))
        sns.heatmap(torch.abs(server.user_context[client_id]['state_dict']['item_commonality.weight'][:, :]))

        wandb.log({
            'commonality.weight': wandb.Image(common_figure),
            'first_personal.weight': wandb.Image(client_personal_fig),
            'first_commonality.weight': wandb.Image(client_common_fig)},
            step=communication_round)

        # update learning rate decay
        server.args['lr_network'] = server.args['lr_network'] * server.args['decay_rate']
        server.args['lr_args'] = server.args['lr_args'] * server.args['decay_rate']

        # update actor config for next round
        for actor in server.ray_actor_list:
            ray.get(actor.sync_args.remote(server.args))

        save_path = server.args['log_dir'] / f"{communication_round}" / f"{communication_round}.pth"

        server.train_data = ray.get(train_data_future)
        plt.close('all')

    # save key information in the round
    if args['save']:
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save(
                {
                    "aggregate_params": global_params,
                    "updated_params": copy.deepcopy(server.global_model.state_dict()),
                    "participants": participants,
                    "user_context": server.user_context,
                    "args": server.args,
                    "data": [server.train_data, server.val_data, server.test_data],
                }, save_path
            )
