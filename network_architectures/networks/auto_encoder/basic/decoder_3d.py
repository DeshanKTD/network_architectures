from network_architectures.components.conv_3d_block import ConvBlock3D
import torch
import torch.nn as nn

class Decoder3D(nn.Module):
    def __init__(self, feature_channels=[128, 64, 32], 
                 kernel_size=2, 
                 stride=2, 
               ):
        super(Decoder3D, self).__init__()
        layers = []
        
        for i in range(len(feature_channels)-1):
            in_channels = feature_channels[i]
            out_channels = feature_channels[i+1]
            layers.append(nn.ConvTranspose3d(in_channels, out_channels,kernel_size=kernel_size,
                                               stride=stride))
            layers.append(ConvBlock3D(in_channels=out_channels,
                                      out_channels=out_channels))
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.decoder(x)