from network_architectures.components.conv_3d_block import ConvBlock3D
import torch
import torch.nn as nn

class Discriminator3D(nn.Module):
    def __init__(self, in_channels=2, base_features=64):
        super(Discriminator3D, self).__init__()
        
        self.model = nn.Sequential(
            ConvBlock3D(in_channels,base_features,kernel_size=4, stride=1, padding=1,activation='leaky_relu'),
            ConvBlock3D(base_features, base_features * 2, kernel_size=4, stride=2, padding=1, activation='leaky_relu', use_batch_norm=True),
            ConvBlock3D(base_features * 2, base_features * 4, kernel_size=4, stride=1, padding=1, activation='leaky_relu', use_batch_norm=True),
            ConvBlock3D(base_features * 4, base_features * 8, kernel_size=4, stride=2, padding=1, activation='leaky_relu', use_batch_norm=True),
            ConvBlock3D(base_features * 8, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.model(x)