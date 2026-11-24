"""Offline executable integration test. Synthetic data, no clinical claims."""
import json
import tempfile
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import MambaConfig,MambaForCausalLM
from datasets.dataset import CTDataset
from models.lungmamba_ear import LungMambaEAR,FIELDS
from models.sfc import generate,HEADINGS
from inference.generate import load_checkpoint

class SmokeTokenizer:
    eos_token_id=0
    def encode(self,text,**kwargs):
        return [HEADINGS.index(text[:-2])+1] if text[:-2] in HEADINGS else [5]
    def decode(self,ids,**kwargs):
        return ' '.join(HEADINGS[i-1] if 1<=i<=4 else f'token_{i}' for i in ids)

def run():
    torch.manual_seed(20261)
    dims={k:2 for k in FIELDS}
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        np.save(root/'synthetic.npy',np.random.default_rng(20261).normal(size=(32,128,128)).astype('float32'))
        record={'volume':'synthetic.npy','roi_center':[16,64,64],'report':'synthetic report','fields':{k:[0,1] for k in FIELDS}}
        (root/'manifest.jsonl').write_text(json.dumps(record)+'\n')
        batch=next(iter(DataLoader(CTDataset(root/'manifest.jsonl',dims),batch_size=1)))
        cfg=MambaConfig(hidden_size=16,state_size=4,num_hidden_layers=1,vocab_size=32,expand=1)
        model=LungMambaEAR(MambaForCausalLM(cfg),dims,width=16)
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5)
        x,c,_,targets,masks=batch
        out=model(x,c,torch.tensor([[6,7,0]]),torch.ones(1,3),targets,masks)
        assert torch.isfinite(out['loss'])
        opt.zero_grad(); out['loss'].backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
        model.eval()
        with torch.no_grad(): expected=model.prefix(x,c)[0]
        path=root/'synthetic_checkpoint.pt'
        torch.save({'state':{k:v for k,v in model.state_dict().items() if not k.startswith('decoder.')}},path)
        restored=LungMambaEAR(MambaForCausalLM(cfg),dims,width=16)
        load_checkpoint(restored,path); restored.eval()
        with torch.no_grad(): torch.testing.assert_close(expected,restored.prefix(x,c)[0])
        text=generate(restored,SmokeTokenizer(),x,c,use_sfc=True,max_length=16,section_max=2)
        assert all(h in text for h in HEADINGS)
        print(json.dumps({'status':'passed','caption_loss':float(out['caption_loss'].detach()),'visual_prefix_shape':list(expected.shape),'checkpoint_roundtrip':True,'synthetic_output':text},indent=2))
if __name__=='__main__': run()
