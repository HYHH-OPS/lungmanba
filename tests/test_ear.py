import torch
import pytest
from transformers import MambaConfig,MambaForCausalLM
from models.lungmamba_ear import *
from models.sfc import generate,HEADINGS
from inference.generate import load_checkpoint

def tiny(fab='token'):
    decoder=MambaForCausalLM(MambaConfig(hidden_size=16,state_size=4,num_hidden_layers=1,vocab_size=32,expand=1))
    return LungMambaEAR(decoder,{k:2 for k in FIELDS},fab,width=16)

def test_shapes():
    x=torch.randn(1,1,32,128,128)
    with torch.no_grad():
        a,b=Encoder3D()(x)
        assert a.shape==(1,96,16,16,16)
        assert b.shape==(1,256,8,4,4)
        gl=GLFPB(16); v=gl(a,b,torch.tensor([[0.,127.,127.]]),x.shape[-3:])
        assert v[:,:64].shape==(1,64,16) and v[:,64:].shape==(1,100,16)
        bridge=BidirectionalMamba(16)
        lengths=[]
        hook=bridge.forward_scan.register_forward_pre_hook(lambda m,args:lengths.append(args[0].shape[1]))
        v,q=bridge(v); hook.remove()
        assert lengths==[196] and v.shape==(1,164,16) and q.shape==(1,32,16)
        assert bridge.forward_scan is not bridge.backward_scan

@pytest.mark.parametrize('fab',['token','query'])
def test_train_and_checkpoint(fab,tmp_path):
    model=tiny(fab); model.train()
    before={k:v.clone() for k,v in model.decoder.state_dict().items()}
    targets={k:torch.zeros(1,2) for k in FIELDS}; masks={k:torch.ones(1,2,dtype=torch.bool) for k in FIELDS}
    x=torch.randn(1,1,8,32,32); c=torch.tensor([[4.,16.,16.]])
    out=model(x,c,torch.tensor([[1,2,3]]),torch.ones(1,3),targets,masks)
    out['loss'].backward()
    assert model.gl.local_proj.weight.grad.abs().sum()>0
    assert model.bridge.queries.grad.abs().sum()>0
    assert model.heads['size'].weight.grad.abs().sum()>0
    assert all(p.grad is None for p in model.decoder.parameters())
    torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=1e-5).step()
    assert all(torch.equal(v,model.decoder.state_dict()[k]) for k,v in before.items())
    path=tmp_path/'test.pt'; torch.save({'state':{k:v for k,v in model.state_dict().items() if not k.startswith('decoder.')}},path)
    load_checkpoint(model,path)
    assert model.prefix(x,c)[0].shape==(1,164,16)

def test_masks():
    pred={'size':torch.zeros(2,2,requires_grad=True)}
    target={'size':torch.tensor([[0.,0.],[float('nan'),float('nan')]])}
    masks={'size':torch.tensor([[True,True],[False,False]])}
    loss=field_loss(pred,target,masks); loss.backward()
    assert torch.isfinite(loss) and (pred['size'].grad[0]>0).all()
    assert (pred['size'].grad[1]==0).all()

class Tokenizer:
    eos_token_id=0
    def encode(self,text,**kwargs): return [HEADINGS.index(text[:-2])+1] if text[:-2] in HEADINGS else [5]
    def decode(self,ids,**kwargs): return ' '.join(HEADINGS[i-1] if 1<=i<=4 else 'x' for i in ids)

def test_sfc():
    model=tiny()
    with torch.no_grad(): model.decoder.lm_head.weight.zero_()
    x=torch.randn(1,1,8,32,32); c=torch.tensor([[4.,16.,16.]])
    on=generate(model,Tokenizer(),x,c,True,max_length=32)
    off=generate(model,Tokenizer(),x,c,False,max_length=32)
    assert all(h in on for h in HEADINGS)
    assert off==''
