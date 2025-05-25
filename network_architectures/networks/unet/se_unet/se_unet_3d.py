import torch.nn as nn
from network_architectures.networks.unet.se_unet.se_encoder_3d import UNetSEEncoder3D
from network_architectures.networks.unet.se_unet.se_decoder_3d import UNetSEDecoder3D



class SEUNet3D(nn.Module):
    def __init__(self, in_channels=1,out_channels=1, feature_channels=[8, 16, 32, 64]):
        super(SEUNet3D, self).__init__()
        
        self.encoder = UNetSEEncoder3D(in_channels=in_channels, feature_channels=feature_channels)
        self.bottleneck = nn.Conv3d(in_channels=feature_channels[-1], out_channels=feature_channels[-1]*2, kernel_size=1)
        self.decoder = UNetSEDecoder3D(feature_channels=feature_channels[::-1])
        self.final_conv = nn.Conv3d(in_channels=feature_channels[0], out_channels=out_channels, kernel_size=1)
        
    def forward(self, x):
        x, skips = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skips)
        x = self.final_conv(x)
        return x