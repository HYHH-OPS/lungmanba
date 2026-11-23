"""Independent Methods-based reference implementation; not original experiment code."""
import torch
from torch import nn
from torch.nn import functional as F
FIELDS = ('location', 'count', 'size', 'density', 'recommendation')

class Encoder3D(nn.Module):
    def __init__(self):
        super().__init__()
        channels = [1,24,48,96,192,256]
        strides = [(1,2,2),(1,2,2),(2,2,2),(2,2,2),(1,2,2)]
        self.stages = nn.ModuleList([nn.Sequential(nn.Conv3d(a,b,3,stride=s,padding=1), nn.InstanceNorm3d(b), nn.LeakyReLU(.01)) for a,b,s in zip(channels,channels[1:],strides)])
    def forward(self,x):
        features=[]
        for layer in self.stages:
            x=layer(x); features.append(x)
        return features[2],features[4]

class GLFPB(nn.Module):
    def __init__(self,width=2560):
        super().__init__()
        self.global_proj=nn.Linear(256,width)
        self.local_proj=nn.Linear(96,width)
    def forward(self,s3,s5,centers,input_shape):
        # Centers are annotated input voxel coordinates in z,y,x order.
        shape=torch.tensor(input_shape,device=centers.device)
        if centers.shape != (s3.shape[0],3) or not torch.isfinite(centers).all() or ((centers<0)|(centers>=shape)).any():
            raise ValueError('Invalid annotated ROI centers')
        # Conv3d padding=1 gives receptive-field centers at multiples of stride.
        coords=torch.floor(centers/centers.new_tensor([2,8,8])).long()
        padded=F.pad(s3,(5,5,2,2,1,1),mode='replicate')
        crops=[]
        for i,(z,y,x) in enumerate(coords.tolist()):
            crops.append(padded[i,:,z:z+2,y:y+5,x:x+10])
        local=torch.stack(crops).flatten(2).transpose(1,2)
        glob=F.adaptive_avg_pool3d(s5,(4,4,4)).flatten(2).transpose(1,2)
        return torch.cat([self.global_proj(glob),self.local_proj(local)],1)

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

def field_loss(logits,targets,masks):
    total=next(iter(logits.values())).sum()*0
    for key,pred in logits.items():
        mask=masks[key].to(pred.dtype).expand_as(pred)
        target=torch.where(mask.bool(),targets[key],torch.zeros_like(targets[key]))
        loss=F.binary_cross_entropy_with_logits(pred,target,reduction='none')
        total=total+(loss*mask).sum()/mask.sum().clamp_min(1)
    return total

class LungMambaEAR(nn.Module):
    def __init__(self,decoder,field_dims,fab='token',width=2560):
        super().__init__()
        if fab not in ('token','query','none'): raise ValueError('Invalid FAB')
        if set(field_dims)!=set(FIELDS): raise ValueError('Five field dimensions required')
        self.decoder=decoder.requires_grad_(False).eval()
        self.encoder=Encoder3D(); self.gl=GLFPB(width); self.bridge=BidirectionalMamba(width)
        self.heads=nn.ModuleDict({k:nn.Linear(width,field_dims[k]) for k in FIELDS})
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
        return {'loss':out.loss+.02*aux,'caption_loss':out.loss,'field_loss':aux}

def load_backbone(path):
    from transformers import AutoTokenizer, MambaForCausalLM
    tokenizer=AutoTokenizer.from_pretrained(path)
    tokenizer.pad_token=tokenizer.eos_token
    return MambaForCausalLM.from_pretrained(path),tokenizer
