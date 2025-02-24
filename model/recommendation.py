import copy
import torch

class PFedRecMo(torch.nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        assert all(key in args for key in ['num_items', 'item_hidden_dim']), f"Missing keys in args: {args.keys()}"
        self.args = args
        self.num_items = args['num_items']
        self.item_hidden_dim = args['item_hidden_dim']

        self.item_commonality = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.item_hidden_dim)

        self.user_embedding = torch.nn.Linear(in_features=self.item_hidden_dim, out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, item_indices):
        if not isinstance(item_indices, torch.Tensor):
            item_indices = torch.tensor(item_indices, dtype=torch.long)
        item_commonality = self.item_commonality(item_indices)

        logits = self.user_embedding(item_commonality)
        rating = self.logistic(logits)

        return rating, None, item_commonality

    def setItemCommonality(self, item_commonality):
        self.item_commonality = copy.deepcopy(item_commonality)
        self.item_commonality.freeze = True

    def commonality_forward(self, item_indices):
        if not isinstance(item_indices, torch.Tensor):
            item_indices = torch.tensor(item_indices, dtype=torch.long)
        item_commonality = self.item_commonality(item_indices)
        logits = self.user_embedding(item_commonality)

        rating = self.logistic(logits)
        return rating, item_commonality

class FedRAPMo(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        assert all(key in args for key in ['num_items', 'item_hidden_dim']), f"Missing keys in args: {args.keys()}"
        self.args = args
        self.num_items = args['num_items']
        self.item_hidden_dim = args['item_hidden_dim']

        self.item_personality = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.item_hidden_dim)
        self.item_commonality = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.item_hidden_dim)

        self.user_embedding = torch.nn.Linear(in_features=self.item_hidden_dim, out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def setItemCommonality(self, item_commonality):
        self.item_commonality = copy.deepcopy(item_commonality)
        self.item_commonality.freeze = True

    def forward(self, item_indices):
        if not isinstance(item_indices, torch.Tensor):
            item_indices = torch.tensor(item_indices, dtype=torch.long)
        item_personality = self.item_personality(item_indices)
        item_commonality = self.item_commonality(item_indices)

        logits = self.user_embedding(item_personality + item_commonality)
        rating = self.logistic(logits)

        return rating, item_personality, item_commonality

    def commonality_forward(self, item_indices):
        if not isinstance(item_indices, torch.Tensor):
            item_indices = torch.tensor(item_indices, dtype=torch.long)
        item_commonality = self.item_commonality(item_indices)
        logits = self.user_embedding(item_commonality)

        rating = self.logistic(logits)
        return rating, item_commonality

class FedPORMo(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        assert all(key in args for key in ['num_items', 'item_hidden_dim']), f"Missing keys in args: {args.keys()}"
        self.args = args
        self.num_items = args['num_items']
        self.item_hidden_dim = args['item_hidden_dim']

        self.item_personality = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.item_hidden_dim)
        self.item_personality.weight.data = torch.nn.functional.normalize(self.item_personality.weight.data, p=2, dim=1)

        self.item_commonality = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.item_hidden_dim)

        self.user_embedding = torch.nn.Linear(in_features=self.item_hidden_dim, out_features=1)
        self.logistic = torch.nn.Sigmoid()

        def normalize_embedding_backward(module, grad_input, grad_output):
            weight = module.weight
            weight.data = weight.data / weight.data.norm(dim=1, keepdim=True)
        self.item_personality.register_full_backward_hook(normalize_embedding_backward)

    def setItemCommonality(self, item_commonality):
        self.item_commonality = copy.deepcopy(item_commonality)
        self.item_commonality.freeze = True

    def forward(self, item_indices):
        if not isinstance(item_indices, torch.Tensor):
            item_indices = torch.tensor(item_indices, dtype=torch.long)
        item_personality = self.item_personality(item_indices)
        item_commonality = self.item_commonality(item_indices)

        item_scale = torch.norm(item_commonality, p=2, dim=1, keepdim=True)
        item_direction = torch.nn.functional.normalize(self.args['gamma'] *item_personality + item_commonality, p=2, dim=1)

        logits = self.user_embedding(item_direction * item_scale)
        rating = self.logistic(logits)

        return rating, item_scale, item_direction

    def commonality_forward(self, item_indices):
        if not isinstance(item_indices, torch.Tensor):
            item_indices = torch.tensor(item_indices, dtype=torch.long)
        item_commonality = self.item_commonality(item_indices)
        logits = self.user_embedding(item_commonality)

        rating = self.logistic(logits)
        return rating, item_commonality

class FedPORAMo(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        assert all(key in args for key in ['num_items', 'item_hidden_dim']), f"Missing keys in args: {args.keys()}"
        self.args = args
        self.num_items = args['num_items']
        self.item_hidden_dim = args['item_hidden_dim']

        self.item_personality = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.item_hidden_dim)
        self.item_commonality = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.item_hidden_dim)

        self.user_embedding = torch.nn.Linear(in_features=self.item_hidden_dim, out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def setItemCommonality(self, item_commonality):
        self.item_commonality = copy.deepcopy(item_commonality)
        self.item_commonality.freeze = True

    def forward(self, item_indices):
        if not isinstance(item_indices, torch.Tensor):
            item_indices = torch.tensor(item_indices, dtype=torch.long)
        item_personality = self.item_personality(item_indices)
        item_commonality = self.item_commonality(item_indices)

        item_scale = torch.norm(item_commonality, p=2, dim=1, keepdim=True)
        item_direction = torch.nn.functional.normalize(self.args['gamma'] * item_personality + item_commonality, p=2, dim=1)

        logits = self.user_embedding(item_direction * item_scale)
        rating = self.logistic(logits)

        return rating, item_scale, item_direction

    def commonality_forward(self, item_indices):
        if not isinstance(item_indices, torch.Tensor):
            item_indices = torch.tensor(item_indices, dtype=torch.long)
        item_commonality = self.item_commonality(item_indices)
        logits = self.user_embedding(item_commonality)

        rating = self.logistic(logits)
        return rating, item_commonality