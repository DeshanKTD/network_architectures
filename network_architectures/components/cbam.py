import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention3D(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels// reduction, in_channels, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        b, c, _, _, _ = x.size()
        avg_out = self.avg_pool(x).view(b,c)
        max_out = self.max_pool(x).view(b,c)
        
        avg_att = self.mlp(avg_out)
        max_att = self.mlp(max_out)
        
        scale = self.sigmoid(avg_att + max_att).view(b, c, 1, 1, 1)
        return x * scale.expand_as(x)
    
class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention3D, self).__init__()
        
        padding = kernel_size // 2
        self.conv = nn.Conv3d(2,1, kernel_size=kernel_size, padding=padding,bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        combined = torch.cat([avg_out, max_out], dim=1)
        combined = self.conv(combined)
        attn = self.sigmoid(combined)
        
        return x * attn.expand_as(x)
    
    
class CBAM3D(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM3D, self).__init__()
        self.chanel_attention = ChannelAttention3D(in_channels, reduction)
        self.spatial_attention = SpatialAttention3D(kernel_size)
        
    def forward(self, x):
        x = self.chanel_attention(x)
        x = self.spatial_attention(x)
        return x