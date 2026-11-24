import argparse
from pathlib import Path
import random
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from datasets.dataset import CTDataset
from models.lungmamba_ear import LungMambaEAR,load_backbone

def main():
    p=argparse.ArgumentParser()
    for k in ('config','train','validation','field-schema','output'): p.add_argument('--'+k,required=True)
    p.add_argument('--base-model',default='state-spaces/mamba-2.8b-hf')
    p.add_argument('--seed',type=int,default=20261)
    a=p.parse_args(); cfg=yaml.safe_load(Path(a.config).read_text()); dims=yaml.safe_load(Path(a.field_schema).read_text())
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise RuntimeError('Training requires BF16-capable CUDA')
    decoder,tok=load_backbone(a.base_model)
    model=LungMambaEAR(decoder,dims,cfg['fab'],auxiliary_coefficient=cfg['auxiliary_coefficient']).cuda()
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    loaders=[DataLoader(CTDataset(m,dims),batch_size=cfg['batch_size'],shuffle=i==0) for i,m in enumerate((a.train,a.validation))]
    if any(len(l.dataset)==0 for l in loaders): raise ValueError('Empty dataset')
    best=float('inf'); Path(a.output).mkdir(parents=True,exist_ok=True)
    for epoch in range(cfg['epochs']):
        for phase,loader in enumerate(loaders):
            model.train(phase==0); total=0.; count=0
            for ct,centers,reports,targets,masks in loader:
                # Explicit EOS is part of the caption target, padding is not.
                seq=[tok.encode(r,add_special_tokens=False)+[tok.eos_token_id] for r in reports]
                tokens=tok.pad({'input_ids':seq},padding=True,return_tensors='pt')
                with torch.set_grad_enabled(phase==0),torch.autocast('cuda',dtype=torch.bfloat16):
                    out=model(ct.cuda(),centers.cuda(),tokens.input_ids.cuda(),tokens.attention_mask.cuda(),{k:v.cuda() for k,v in targets.items()},{k:v.cuda() for k,v in masks.items()})
                if phase==0:
                    optimizer.zero_grad(); out['loss'].backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip']); optimizer.step()
                n=int(tokens.attention_mask.sum()); total+=out['caption_loss'].item()*n; count+=n
            if phase==1 and total/count<best:
                best=total/count
                state={k:v.cpu() for k,v in model.state_dict().items() if not k.startswith('decoder.')}
                torch.save({'state':state,'field_dims':dims,'fab':cfg['fab'],'epoch':epoch,'validation_nll':best},Path(a.output)/'best.pt')
        print(f'Epoch {epoch+1}: best validation caption NLL {best:.6f}')
if __name__=='__main__': main()
