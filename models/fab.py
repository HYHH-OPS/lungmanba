import torch
from torch import nn
from torch.nn import functional as F

def field_loss(logits,targets,masks):
    total=next(iter(logits.values())).sum()*0
    for key,pred in logits.items():
        mask=masks[key].to(pred.dtype).expand_as(pred)
        target=torch.where(mask.bool(),targets[key],torch.zeros_like(targets[key]))
        loss=F.binary_cross_entropy_with_logits(pred,target,reduction='none')
        total=total+(loss*mask).sum()/mask.sum().clamp_min(1)
    return total

