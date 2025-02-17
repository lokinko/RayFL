import torch

class FedRAPLoss(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.crit = torch.nn.BCELoss()
        self.independency = torch.nn.MSELoss()

        if self.args['regular'] == 'l1':
            self.reg = torch.nn.L1Loss()
        else:
            self.reg = torch.nn.MSELoss()

    def forward(self, ratings_pred, ratings, item_personality, item_commonality):
        if self.args['regular'] == 'none':
            self.args['mu'] = 0

        if self.args['regular'] == 'nuc':
            third = torch.norm(item_commonality, p='nuc')

        elif self.args['regular'] == 'inf':
            third = torch.norm(item_commonality, p=float('inf'))

        else:
            dummy_target = torch.zeros_like(item_commonality, requires_grad=False)
            third = self.reg(item_commonality, dummy_target)

        loss = self.crit(ratings_pred, ratings) \
               - self.args['lambda'] * self.independency(item_personality, item_commonality) \
               + self.args['mu'] * third

        return loss
