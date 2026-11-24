import torch
from torch import nn
from torch.nn import functional as F

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
