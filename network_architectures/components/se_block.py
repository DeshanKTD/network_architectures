import torch.nn as nn

# Channel Squeeze and Excitation (SE) Block
class ChannelSEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelSEBlock, self).__init__()
        
        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.squeeze(x).view(b,c)
        y = self.excitation(y).view(b,c,1,1,1)
        return x * y.expand_as(x)
    
# Spatial Squeeze and Excitation (SE) Block
class SpatialSEBlock(nn.Module):
    def __init__(self, channels):
        super(SpatialSEBlock, self).__init__()
        
        self.conv = nn.Conv3d(in_channels=channels, out_channels=1, kernel_size =1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        y = self.conv(x)
        y = self.sigmoid(y)
        return x * y.expand_as(x)
    
# Channel and Spatial Squeeze and Excitation (SE) Block
class ChannelSpatialSEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelSpatialSEBlock, self).__init__()
        
        self.channel_se = ChannelSEBlock(channels, reduction)
        self.spatial_se = SpatialSEBlock(channels)
        
    def forward(self, x):
        cse_out = self.channel_se(x)
        sse_out = self.spatial_se(x)
        return cse_out + sse_out
        
    
        