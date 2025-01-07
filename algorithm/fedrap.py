import copy
import logging

import ray

from core.server.fedrap_server import FedRapServer

def run(args):
    server = FedRapServer(args)
    server.allocate_init_status()
    logging.info(f"Creates {args['method']} server successfully.")

    for communication_round in range(1, args['num_rounds']+1):
        logging.info(f"{'='*20} Round {communication_round} starts {'='*20}")
        participants = server.select_participants()

        train_data = server.dataset.sample_train_data.remote()
        server.train_on_round(participants)
        round_loss = sum(sum(server.users[user_id]['loss']) / len(server.users[user_id]['loss']) for user_id in participants) / len(participants)

        origin_params = copy.deepcopy(server.model.state_dict())
        server_params = server.aggregate(participants)

        for _, user in server.users.items():
            user['model_dict'].update(server_params)

        server.model.load_state_dict(server.model.state_dict() | server_params)

        val_hr, val_ndcg = server.test_on_round(server.val_data)
        test_hr, test_ndcg = server.test_on_round(server.test_data)
        logging.info(
            f"Round = {communication_round}, Loss = {round_loss:.6f} "
            f"Val HR = {val_hr:.4f}, Val NDCG = {val_ndcg:.4f}, Test HR = {test_hr:.4f}, Test NDCG = {test_ndcg:.4f}")

        server.args['lr_network'] = server.args['lr_network'] * server.args['decay_rate']
        server.args['lr_args'] = server.args['lr_args'] * server.args['decay_rate']

        save_path = server.args['log_dir'] / f"{communication_round}" / f"{communication_round}.pth"

        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)

        # torch.save(
        #     {
        #         "origin_params": origin_params,
        #         "aggregate_params": server_params,
        #         "updated_params": server.model.state_dict(),
        #         "participants": participants,
        #         "users": server.users,
        #         "args": server.args,
        #         "data": [server.train_data, server.val_data, server.test_data],
        #         "metrics": [test_hr, test_ndcg]
        #     }, save_path
        # )

        server.train_data = ray.get(train_data)