import torch
import torch.nn as nn
from torch.nn import functional as F
from .ssim import SSIM

mse_criterion = nn.MSELoss(reduction='none')
class_criterion = nn.BCEWithLogitsLoss(size_average=False).cuda()
BCELoss_criterion = nn.BCELoss()
ssim_criterion = SSIM(window_size=11)




def cross_entropy(**kwargs):
        class_criterion.reduction = 'none'
        class_loss = class_criterion(kwargs['pred'], kwargs['target'])
        class_loss = torch.sum(class_loss, dim=(1, 2, 3))
        class_loss = torch.mean(class_loss)
        return class_loss

def ssim_loss(**kwargs):
        return -ssim_criterion(kwargs['pred'], kwargs['target'])

def mse_loss(**kwargs):
        mse_loss = mse_criterion(kwargs['pred'], kwargs['target'])
        mse_loss = torch.sum(mse_loss, dim=(1, 2, 3))
        mse_loss = torch.mean(mse_loss)
        return mse_loss

def vae_loss(**kwargs):
        recons_loss = F.mse_loss(kwargs['pred'], kwargs['target'])
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + kwargs['log_var'] - kwargs['mu'] ** 2 - kwargs['log_var'].exp(), dim=1), dim=0)
        loss = recons_loss + 0.00001 * kld_loss
        return [loss, recons_loss, -kld_loss]

def BCE_loss(**kwargs):
        BCE_loss = BCELoss_criterion(kwargs['pred'], kwargs['target'])
        return BCE_loss


def BCE_loss(**kwargs):
        BCE_loss = BCELoss_criterion(kwargs['pred'], kwargs['target'])
        return BCE_loss