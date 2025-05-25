import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention3D(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention3D, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.global_avg_pool(x).view(b,c)
        y = self.fc(y).view(b,c,1,1,1)
        return x * y.expand_as(x)
    
    
class SpatialAttention3D(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention3D, self).__init__()
        self.spatial = nn.Sequential(
            nn.Conv3d(in_channels, in_channels//4, kernel_size=1),
            nn.BatchNorm3d(in_channels//4),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels//4,1,kernel_size=3,padding=1),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.spatial(x)
        return y.expand_as(x)
    
class BAM3D(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(BAM3D, self).__init__()
        self.channel_attention = ChannelAttention3D(in_channels, reduction)
        self.spatial_attention = SpatialAttention3D(in_channels)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        channel_out = self.channel_attention(x)
        spatial_out = self.spatial_attention(x)
        attention = self.sigmoid(channel_out + spatial_out)
        return x * attention.expand_as(x)
    
    