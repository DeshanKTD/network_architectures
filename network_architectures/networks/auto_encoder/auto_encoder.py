import torch
import torch.nn as nn

from network_architectures.networks.auto_encoder.decoder_3d import Decoder3D
from network_architectures.networks.auto_encoder.encoder_3d import Encoder3D

class AutoEncoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(AutoEncoder,self).__init__()
        self.encoder = Encoder3D(in_channels=in_channels,feature_channels=[32, 64, 128])
        # Decoder expects the feature channels in reverse order of the encoder and one more addional up-sampling layer
        # to match the output channels with the input channels.
        # The last layer of the decoder will be a Conv3D layer to reduce the channels to the desired output channels.
        self.decoder = Decoder3D(feature_channels=[128, 64, 32, 16])
        self.final_conv = nn.Conv3d(in_channels=16, out_channels=out_channels, kernel_size=1)
        
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.final_conv(x)
        return x
        
        
