import torch
from torch import nn
from torch.nn import functional as F

class BidirectionalMamba(nn.Module):
    def __init__(self,width=2560):
        super().__init__()
        from transformers import MambaConfig, MambaModel
        cfg=MambaConfig(hidden_size=width,state_size=16,num_hidden_layers=1,expand=2,vocab_size=1)
        self.forward_scan=MambaModel(cfg).layers[0].mixer
        self.backward_scan=MambaModel(cfg).layers[0].mixer
        self.queries=nn.Parameter(torch.randn(1,32,width)*.02)
        self.proj=nn.Linear(width*2,width)
        self.norm=nn.LayerNorm(width)
    def forward(self,v):
        x=torch.cat([v,self.queries.expand(v.shape[0],-1,-1)],1)
        f=self.forward_scan(x)
        b=self.backward_scan(x.flip(1)).flip(1)
        x=self.norm(x+self.proj(torch.cat([f,b],-1)))
        return x[:,:164],x[:,164:]

