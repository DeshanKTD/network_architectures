from network_architectures.components.bam import BAM3D
import torch.nn as nn
from network_architectures.networks.unet.basic.basic_unet_decoder import UNetDecoder3D
from network_architectures.networks.unet.basic.basic_unet_encoder import UNetEncoder3D


class BAMUNet3D(nn.Module):
    def __init__(self, in_channels=1,out_channels=1, feature_channels=[8, 16, 32, 64]):
        super(BAMUNet3D, self).__init__()
        
        self.encoder = UNetEncoder3D(in_channels=in_channels, feature_channels=feature_channels)
        self.bottleneck = nn.Conv3d(in_channels=feature_channels[-1], out_channels=feature_channels[-1]*2, kernel_size=1)
        self.decoder = UNetDecoder3D(feature_channels=feature_channels[::-1])
        self.bam = BAM3D(in_channels=feature_channels[-1]*2) # Apply BAM after bottleneck
        self.final_conv = nn.Conv3d(in_channels=feature_channels[0], out_channels=out_channels, kernel_size=1)
        
    def forward(self, x):
        x, skips = self.encoder(x)
        x = self.bottleneck(x)
        x = self.bam(x) # Apply BAM after bottleneck
        x = self.decoder(x, skips)
        x = self.final_conv(x)
        return x