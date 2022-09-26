import torch
import torch.nn.functional as F
import numpy as np

def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    x = torch.sum(score * target)
    y = torch.sum(score * score)
    z = torch.sum(target * target)
    loss = 1 - (2 * x + smooth) / (y + z + smooth)
    return loss

def softmax_mse_loss(score, target):
    score_softmax = F.softmax(score, dim=1)
    target_softmax = F.softmax(target, dim=1)
    loss = (score_softmax - target_softmax) ** 2
    return loss

def sigmod_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))