import argparse
import numpy as np
import torch
from models.lungmamba_ear import LungMambaEAR,load_backbone
from models.sfc import generate

def load_checkpoint(model,path):
    state=torch.load(path,map_location='cpu',weights_only=True)
    result=model.load_state_dict(state['state'],strict=False)
    if result.unexpected_keys or any(not k.startswith('decoder.') for k in result.missing_keys): raise ValueError('Checkpoint architecture mismatch')
    return model

def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--volume',required=True)
    p.add_argument('--roi-center',nargs=3,type=float,required=True)
    p.add_argument('--base-model',default='state-spaces/mamba-2.8b-hf'); p.add_argument('--no-sfc',action='store_true')
    a=p.parse_args(); state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    decoder,tok=load_backbone(a.base_model)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=load_checkpoint(LungMambaEAR(decoder,state['field_dims'],state['fab']),a.checkpoint).to(device)
    x=np.load(a.volume,allow_pickle=False)
    if x.shape!=(32,128,128): raise ValueError('Expected preprocessed volume')
    print(generate(model,tok,torch.tensor(x,dtype=torch.float32,device=device)[None,None],torch.tensor([a.roi_center],device=device),use_sfc=not a.no_sfc))
if __name__=='__main__': main()
