import torch
import torch.nn as nn
import torch.nn.functional as F

class PyramidPooling3D(nn.Module):
    def __init__(self, in_channels, pool_sizes=[1,2,3,6]):
        super(PyramidPooling3D, self).__init__()
        
        out_channels = in_channels // len(pool_sizes)
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool3d(output_size=(ps,ps,ps)),
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.ReLU(inplace=True)
            ) for ps in pool_sizes
        ])
        self.bottleneck = nn.Conv3d(in_channels + out_channels * len(pool_sizes),in_channels,kernel_size=1)
        
    def forward(self, x):
        d, h, w = x.shape[2:]
        pooled_features = [x]
        
        for stage in self.stages:
            out = stage(x)
            out = F.interpolate(out, size=(d,h,w), mode='trilinear', align_corners=True)
            pooled_features.append(out)
            
        out = torch.cat(pooled_features, dim=1)
        out = self.bottleneck(out)
        return out
        
        