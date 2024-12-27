import torch

from core.model.model.collaborate_filter import PersonalizedCollaboFilterModel

def build_model(args) -> torch.nn.Module:
    if args['method'] == 'fedrap' and args['model'] == 'cf':
        model = PersonalizedCollaboFilterModel(args)
    else:
        raise NotImplementedError(f"Model {args['model']} not implemented")
    return model