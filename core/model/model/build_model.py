import torch

from core.model.model.collaborate_filter import CollaborativeFilterModel

def build_model(args) -> torch.nn.Module:
    if args['model'] == 'cf':
        model = CollaborativeFilterModel(args)
    else:
        raise NotImplementedError(f"Model {args['model']} not implemented")
    return model