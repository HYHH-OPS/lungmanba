import torch
from torch import nn
from torch.nn import functional as F

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

