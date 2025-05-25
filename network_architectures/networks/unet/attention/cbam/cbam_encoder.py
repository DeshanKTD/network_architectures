from network_architectures.components.cbam import CBAM3D
import torch
import torch.nn as nn

from network_architectures.components.conv_3d_block import ConvBlock3D
from network_architectures.components.se_block import ChannelSpatialSEBlock



class CBAMConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CBAMConvBlock3D, self).__init__()
        self.block = nn.Sequential(
            ConvBlock3D(in_channels, out_channels),
            CBAM3D(out_channels)
        )

    def forward(self, x):
        return self.block(x)


class UNetCBAMEncoder3D(nn.Module):
    def __init__(self, in_channels=1, feature_channels=[8,16,32,64]):
        super(UNetCBAMEncoder3D, self).__init__()
        
        self.blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        prev_channels = in_channels
        
        for out_channels in feature_channels:
            self.blocks.append(CBAMConvBlock3D(in_channels=prev_channels,
                                          out_channels=out_channels))
            self.pools.append(nn.MaxPool3d(kernel_size=2, stride=2))
            prev_channels = out_channels
            
            
    def forward(self,x):
        skips = []
        for block, pool in zip(self.blocks, self.pools):
            x = block(x)
            skips.append(x)
            x = pool(x)
        return x, skips 