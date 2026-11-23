"""Independent Methods-based reference implementation; not original experiment code."""
import torch
from torch import nn
from torch.nn import functional as F
FIELDS = ('location', 'count', 'size', 'density', 'recommendation')

from .encoder_3d import Encoder3D
from .gl_fpb import GLFPB
from .bidirectional_mamba import BidirectionalMamba
from .fab import field_loss

class LungMambaEAR(nn.Module):
    def __init__(self,decoder,field_dims,fab='token',width=2560,auxiliary_coefficient=0.02):
        super().__init__()
        if fab not in ('token','query','none'): raise ValueError('Invalid FAB')
        if set(field_dims)!=set(FIELDS): raise ValueError('Five field dimensions required')
        self.decoder=decoder.requires_grad_(False).eval()
        self.encoder=Encoder3D(); self.gl=GLFPB(width); self.bridge=BidirectionalMamba(width)
        self.heads=nn.ModuleDict({k:nn.Linear(width,field_dims[k]) for k in FIELDS})
        if decoder.get_input_embeddings().embedding_dim != width: raise ValueError("Decoder and bridge widths must match")
        self.auxiliary_coefficient=auxiliary_coefficient
        self.fab=fab
    def train(self,mode=True):
        super().train(mode); self.decoder.eval(); return self
    def prefix(self,ct,centers):
        return self.bridge(self.gl(*self.encoder(ct),centers,ct.shape[-3:]))
    def forward(self,ct,centers,ids,attention,targets,masks):
        v,q=self.prefix(ct,centers)
        emb=self.decoder.get_input_embeddings()(ids)
        inputs=torch.cat([v.to(emb.dtype),emb],1)
        labels=ids.masked_fill(~attention.bool(),-100)
        labels=torch.cat([labels.new_full((ids.shape[0],164),-100),labels],1)
        out=self.decoder(inputs_embeds=inputs,labels=labels,use_cache=False)
        pooled=(v if self.fab!='query' else q).mean(1)
        logits={k:h(pooled) for k,h in self.heads.items()}
        aux=field_loss(logits,targets,masks) if self.fab!='none' else v.sum()*0
        return {'loss':out.loss+self.auxiliary_coefficient*aux,'caption_loss':out.loss,'field_loss':aux}

def load_backbone(path):
    from transformers import AutoTokenizer, MambaForCausalLM
    tokenizer=AutoTokenizer.from_pretrained(path)
    tokenizer.pad_token=tokenizer.eos_token
    return MambaForCausalLM.from_pretrained(path),tokenizer
