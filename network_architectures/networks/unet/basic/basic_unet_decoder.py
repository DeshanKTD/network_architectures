import torch
import torch.nn as nn
from network_architectures.components.conv_3d_block import ConvBlock3D


class UNetDecoder3D(nn.Module):
    def __init__(self, feature_channels=[64, 32, 16, 8], kernel_size=2, stride=2):
        super(UNetDecoder3D, self).__init__()
        
        self.upconvs = nn.ModuleList()
        self.blocks = nn.ModuleList()
        
        for i in range(len(feature_channels)):
            if i == 0:
                # The first upconv does not have a skip connection
                in_channels = feature_channels[i]*2
                out_channels = feature_channels[i]
            else:
                in_channels = feature_channels[i-1]
                out_channels = feature_channels[i]
            self.upconvs.append(nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride))
            self.blocks.append(ConvBlock3D(in_channels=out_channels*2, out_channels=out_channels, use_batch_norm=True))
        
    def forward(self, x, skips):
        for i in range(len(self.upconvs)):  # run 0
            x = self.upconvs[i](x)          # start with 64 -> 32
            skip = skips[-(i + 1)]          # extract 32
            x = torch.cat((x, skip), dim=1) # results 
            x = self.blocks[i](x)
        return x
    
    
    