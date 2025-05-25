from network_architectures.components.conv_3d_block import ConvBlock3D
import torch
import torch.nn as nn

class UNetEncoder3D(nn.Module):
    def __init__(self, in_channels=1, feature_channels=[8,16,32,64]):
        super(UNetEncoder3D, self).__init__()
        
        self.blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        prev_channels = in_channels
        
        for out_channels in feature_channels:
            self.blocks.append(ConvBlock3D(in_channels=prev_channels,
                                          out_channels=out_channels,
                                          use_batch_norm=True))
            self.pools.append(nn.MaxPool3d(kernel_size=2, stride=2))
            prev_channels = out_channels
            
            
    def forward(self,x):
        skips = []
        for block, pool in zip(self.blocks, self.pools):
            x = block(x)
            skips.append(x)
            x = pool(x)
        return x, skips 