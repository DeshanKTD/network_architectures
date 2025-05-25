from network_architectures.components.conv_3d_block import ConvBlock3D
import torch
import torch.nn as nn

class Encoder3D(nn.Module):
    def __init__(self, in_channels, feature_channels=[32,64,128]):
        super(Encoder3D, self).__init__()
        
        layers = []
        
        prev_channels = in_channels
        for out_channels in feature_channels:
            layers.append(ConvBlock3D(in_channels=prev_channels, 
                                      out_channels=out_channels,use_batch_norm=True,))
            layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
            prev_channels = out_channels
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.encoder(x)
           