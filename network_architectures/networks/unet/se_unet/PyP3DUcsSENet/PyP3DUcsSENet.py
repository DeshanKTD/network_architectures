import torch.nn as nn
from network_architectures.networks.unet.se_unet.se_encoder_3d import UNetSEEncoder3D
from network_architectures.networks.unet.se_unet.se_decoder_3d import UNetSEDecoder3D
from network_architectures.components.pyramid_pooling import PyramidPooling3D


# Akiz's Network
# PyP3DUcsSENet: Pyramid Pooling 3D U-Net with Channel and Spatial Squeeze-and-Excitation (SE) blocks
# This network architecture combines a U-Net structure with pyramid pooling and SE blocks for enhanced feature extraction and representation.

# Reference:
# Rahman, M.A., Singh, S., Shanmugalingam, K., Iyer, S., Blair, A., Ravindran, P. and Sowmya, A., 2023, November. 
# Attention and Pooling based Sigmoid Colon Segmentation in 3D CT images. 
# In 2023 International Conference on Digital Image Computing: Techniques 
# and Applications (DICTA) (pp. 312-319). IEEE
class PyP3DUcsSENet(nn.Module):
    def __init__(self, in_channels=1,out_channels=1, feature_channels=[8, 16, 32, 64],pool_sizes=[1,2,4,8]):
        super(PyP3DUcsSENet, self).__init__()
        
        self.encoder = UNetSEEncoder3D(in_channels=in_channels, feature_channels=feature_channels)
        self.bottleneck = nn.Conv3d(in_channels=feature_channels[-1], out_channels=feature_channels[-1]*2, kernel_size=1)
        self.pyramid_pooling = PyramidPooling3D(in_channels=feature_channels[-1]*2,pool_sizes=pool_sizes)
        self.decoder = UNetSEDecoder3D(feature_channels=feature_channels[::-1])
        self.final_conv = nn.Conv3d(in_channels=feature_channels[0], out_channels=out_channels, kernel_size=1)
        
    def forward(self, x):
        x, skips = self.encoder(x)
        x = self.bottleneck(x)
        x = self.pyramid_pooling(x)
        x = self.decoder(x, skips)
        x = self.final_conv(x)
        return x