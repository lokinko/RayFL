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

        train_data_future = server.dataset.sample_federated_train_data.remote()
        server.train_on_round(participants)

        round_loss = sum(sum(server.user_context[user_id]['loss']) / len(server.user_context[user_id]['loss']) for user_id in participants) / len(participants)

        origin_params = copy.deepcopy(server.global_model.state_dict())
        updated_params = server.aggregate(participants, ['item_commonality.weight'])

        for _, user_context in server.user_context.items():
            user_context['state_dict']['item_commonality.weight'] = copy.deepcopy(
                                updated_params['item_commonality.weight'])

        server_params = copy.deepcopy(server.global_model.state_dict())
        server_params['item_commonality.weight'] = updated_params['item_commonality.weight'].data
        server.global_model.load_state_dict(server_params)

        val_hr, val_ndcg = server.test_on_round(server.val_data)
        test_hr, test_ndcg = server.test_on_round(server.test_data)
        com_hr, com_ndcg = server.test_commonality(server.test_data)
        decouple_hr, decouple_ndcg, avg_item_personal = server.test_decouple(participants, server.test_data)

        common_figure = plt.figure(figsize=(10, 10))
        sns.heatmap(torch.abs(updated_params['item_commonality.weight'][:100, :]))

        average_figure = plt.figure(figsize=(10, 10))
        sns.heatmap(torch.abs(avg_item_personal['item_personality.weight'][:100, :]))

        first_client_personal_figure = plt.figure(figsize=(10, 10))
        sns.heatmap(torch.abs(server.user_context[0]['state_dict']['item_personality.weight'][:100, :]))

        first_client_common_figure = plt.figure(figsize=(10, 10))
        sns.heatmap(torch.abs(server.user_context[0]['state_dict']['item_commonality.weight'][:100, :]))

        logging.info(
            f"Val HR = {val_hr:.4f}, Val NDCG = {val_ndcg:.4f}, Test HR = {test_hr:.4f}, Test NDCG = {test_ndcg:.4f}, "
            f"Commonality HR = {com_hr:.4f}, Commonality NDCG = {com_ndcg:.4f}, "
            f"Decouple HR = {decouple_hr:.4f}, Decouple NDCG = {decouple_ndcg:.4f}"
        )

        wandb.log({
            'commonality.weight': wandb.Image(common_figure),
            'avg_personal.weight': wandb.Image(average_figure),
            'first_personal.weight': wandb.Image(first_client_personal_figure),
            'first_commonality.weight': wandb.Image(first_client_common_figure),

            'round_loss': round_loss,
            'val_hr': val_hr, 'val_ndcg': val_ndcg,
            'test_hr': test_hr, 'test_ndcg': test_ndcg,
            'commonality_hr': com_hr, 'commonality_ndcg': com_ndcg,
            'decouple_hr': decouple_hr, 'decouple_ndcg': decouple_ndcg},
            step=communication_round)

        server.args['lr_network'] = server.args['lr_network'] * server.args['decay_rate']
        server.args['lr_args'] = server.args['lr_args'] * server.args['decay_rate']

        save_path = server.args['log_dir'] / f"{communication_round}" / f"{communication_round}.pth"


        if args['save']:
            if not save_path.parent.exists():
                save_path.parent.mkdir(parents=True, exist_ok=True)

                torch.save(
                    {
                        "origin_params": origin_params,
                        "aggregate_params": updated_params,
                        "updated_params": copy.deepcopy(server.global_model.state_dict()),
                        "averaged_item_personal": avg_item_personal,

                        "participants": participants,
                        "user_context": server.user_context,
                        "args": server.args,
                        "data": [server.train_data, server.val_data, server.test_data],
                    }, save_path
                )

        server.train_data = ray.get(train_data_future)
        plt.close('all')
