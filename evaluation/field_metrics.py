"""Micro F1 over available binary field annotations, not a report parser."""
import torch
def score(prediction,target,mask):
    valid=mask.bool().expand_as(target)
    p=prediction.bool()[valid]; t=target.bool()[valid]
    tp=int((p&t).sum()); fp=int((p&~t).sum()); fn=int((~p&t).sum())
    denominator=2*tp+fp+fn
    return {'tp':tp,'fp':fp,'fn':fn,'available':int(valid.sum()),'f1':2*tp/denominator if denominator else None}
