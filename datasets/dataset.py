import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from models.lungmamba_ear import FIELDS
class CTDataset(Dataset):
    def __init__(self,manifest,field_dims):
        self.root=Path(manifest).resolve().parent
        self.rows=[json.loads(s) for s in Path(manifest).read_text().splitlines() if s.strip()]
        self.dims=field_dims
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        x=np.load(self.root/r['volume'],allow_pickle=False)
        if x.shape!=(32,128,128) or not np.isfinite(x).all(): raise ValueError('Expected finite preprocessed 32x128x128 array')
        targets={}; masks={}
        for k in FIELDS:
            value=r.get('fields',{}).get(k)
            targets[k]=torch.tensor(value if value is not None else [0.]*self.dims[k],dtype=torch.float32)
            if targets[k].shape!=(self.dims[k],): raise ValueError('Field vocabulary mismatch')
            masks[k]=torch.full((self.dims[k],),value is not None)
        return torch.from_numpy(x.astype('float32')).unsqueeze(0),torch.tensor(r['roi_center'],dtype=torch.float32),r['report'],targets,masks
