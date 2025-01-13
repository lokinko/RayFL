import torch

from core.model.model.collaborate_filter import PersonalizedCollaboFilterModel, CollaboFilterModel
from core.model.model.seq_decoder import CustomGPT2LMHeadModel

def build_model(args) -> torch.nn.Module:
    if args['method'] == 'fedrap' and args['model'] == 'pcf':
        model = PersonalizedCollaboFilterModel(args)
    elif args['method'] == "fedncf" and args['model'] == 'cf':
        model = CollaboFilterModel(args)
    elif args['method'] == "fedaug":
        from transformers import GPT2Config
        gpt2_config = GPT2Config(
                vocab_size=args['num_items'],
                n_layer=1,
                n_head=1,
                n_embd=32,
                max_position_embeddings=5,
                # max_position_embeddings=args['seq_length'] if args['condational_method']=='add' else args['seq_length'] + 1 ,
        )
        # decoder = CustomGPT2LMHeadModel(gpt2_config, condational_method=args['condational_method'])
        decoder = CustomGPT2LMHeadModel(gpt2_config, condational_method='add', num_users=args['num_users'])
        
        if args['model'] == 'pcf':
            model = PersonalizedCollaboFilterModel(args)
        elif args['model'] == 'cf':
            model = CollaboFilterModel(args)
        return decoder, model
    else:
        raise NotImplementedError(f"Model {args['model']} not implemented")
    return model