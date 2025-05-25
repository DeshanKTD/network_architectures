from network_architectures.components.conv_3d_block import ConvBlock3D
import torch
import torch.nn as nn

class VAEEncoder3D(nn.Module):
    def __init__(self, in_channels=1, feature_channels=[32,64,128], latent_dims=256, input_shape=(256, 256, 256)):
        super(VAEEncoder3D, self).__init__()
        layers = []
        
        prev_channels = in_channels
        for out_channels in feature_channels:
            layers.append(ConvBlock3D(in_channels=prev_channels, 
                                      out_channels=out_channels,use_batch_norm=True,))
            layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
            prev_channels = out_channels
        
        self.conv = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        
        flatten_size = (
            feature_channels[-1] * 
            (input_shape[0] // (2 ** len(feature_channels))) * 
            (input_shape[1] // (2 ** len(feature_channels))) * 
            (input_shape[2] // (2 ** len(feature_channels)))
            )
        self.fc_mu = nn.Linear(flatten_size, latent_dims)  # Assuming input size is reduced to 128x4x4x4
        self.fc_logvar = nn.Linear(flatten_size, latent_dims)

        
        
    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    