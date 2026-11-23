"""Parameter-free reference section controller (greedy decoding)."""
import torch
HEADINGS=('Imaging Findings','Conclusion','Recommendation','Pathological Tendency')
@torch.no_grad()
def generate(model,tokenizer,ct,centers,use_sfc=True,max_length=1024,section_max=200):
    if ct.shape[0]!=1: raise ValueError('Generation supports one case at a time')
    if not 1<=max_length<=1024: raise ValueError('max_length must be 1..1024')
    model.eval(); v,_=model.prefix(ct,centers)
    generated=[]
    def next_token(body):
        emb=model.decoder.get_input_embeddings()
        seq=emb(torch.tensor([generated],device=v.device)) if generated else v[:,:0]
        logits=model.decoder(inputs_embeds=torch.cat([v.to(emb.weight.dtype),seq],1),use_cache=False).logits[0,-1].clone()
        # Suppress repeated bigrams within the last 16 body tokens.
        recent=body[-16:]
        if recent:
            for a,b in zip(recent,recent[1:]):
                if a==recent[-1]: logits[b]=-torch.inf
        if use_sfc:
            # Prevent the body from completing any reserved heading.
            for heading in HEADINGS:
                for spelling in (heading, '\n'+heading):
                    ids=tokenizer.encode(spelling+':\n',add_special_tokens=False)
                    if ids and (len(ids)==1 or generated[-(len(ids)-1):]==ids[:-1]):
                        logits[ids[-1]]=-torch.inf
        return int(logits.argmax())
    if not use_sfc:
        while len(generated)<max_length:
            token=next_token([])
            if token==tokenizer.eos_token_id: break
            generated.append(token)
    else:
        headings=[tokenizer.encode(h+':\n',add_special_tokens=False) for h in HEADINGS]
        separator=tokenizer.encode('\n',add_special_tokens=False)
        if sum(map(len,headings))+4*len(separator)>max_length: raise ValueError('Budget too small for headings')
        for section,heading in enumerate(headings):
            generated.extend(heading); body=[]
            reserve=sum(map(len,headings[section+1:]))+(4-section)*len(separator)
            budget=min(section_max,max_length-len(generated)-reserve)
            for _ in range(budget):
                token=next_token(body)
                if token==tokenizer.eos_token_id: break
                body.append(token); generated.append(token)
            generated.extend(separator)
    return tokenizer.decode(generated,skip_special_tokens=True)
