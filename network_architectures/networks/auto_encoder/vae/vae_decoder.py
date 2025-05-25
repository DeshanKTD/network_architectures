import torch
import torch.nn as nn
from network_architectures.components.conv_3d_block import ConvBlock3D


class VAEDecoder(nn.Module):
    def __init__(self, feature_channels=[128,64, 32], latent_dims = 64, initial_shape = (128,4,4,4)):
        super(VAEDecoder, self).__init__()
        
        self.initial_shape = initial_shape
        self.fc = nn.Linear(latent_dims, torch.prod(torch.tensor(initial_shape)))
        
        layers = []
        
        for i in range(len(feature_channels)):
            in_channels = feature_channels[i]
            if(i < len(feature_channels) - 1):
                out_channels = feature_channels[i+1]
            elif(i == len(feature_channels) - 1):
                out_channels = feature_channels[i]
            layers.extend([
                           nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),
                           ConvBlock3D(in_channels=out_channels, out_channels=out_channels)
                           ])
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, *self.initial_shape)
        x = self.decoder(x)
        return x