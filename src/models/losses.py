import torch, torch.nn as nn

class SmoothL1Multi(nn.Module):
    def __init__(self, sbp_weight=1.25):
        super().__init__()
        self.huber = nn.SmoothL1Loss(reduction="none")
        self.sbp_w = sbp_weight
    def forward(self, pred, target):
        # pred, target: [B,2]
        loss = self.huber(pred, target)  # [B,2]
        loss[:,0] = loss[:,0]*self.sbp_w
        return loss.mean()

class HeteroscedasticLoss(nn.Module):
    """Model outputs [mu_sbp, mu_dbp, logvar_sbp, logvar_dbp]"""
    def __init__(self, sbp_weight=1.25):
        super().__init__()
        self.sbp_w = sbp_weight
    def forward(self, pred, target):
        mu = pred[:,:2]
        logv = pred[:,2:]
        inv_var = torch.exp(-logv)
        se = (mu - target)**2
        nll = 0.5*(se*inv_var + logv)
        nll[:,0] = nll[:,0]*self.sbp_w
        return nll.mean()
