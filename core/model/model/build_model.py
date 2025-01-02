import torch

from core.model.model.collaborate_filter import PersonalizedCollaboFilterModel, CollaboFilterModel

def build_model(args) -> torch.nn.Module:
    if args['method'] == 'fedrap' and args['model'] == 'cf':
        model = PersonalizedCollaboFilterModel(args)
    elif args['method'] == "fedncf" and args['model'] == 'cf':
        model = CollaboFilterModel(args)
    else:
        raise NotImplementedError(f"Model {args['model']} not implemented")
    return model