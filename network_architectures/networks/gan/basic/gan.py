

from network_architectures.networks.gan.basic.discriminator import Discriminator3D
from network_architectures.networks.unet.se_unet.se_unet_3d import SEUNet3D


generator = SEUNet3D(in_channels=2, out_channels=1, feature_channels=[8, 16, 32, 64])
discriminator = Discriminator3D(in_channels=2, base_features=8)


# Create a trainer to do the gap filling of colon segmenations

# nnUNet Trainer
# Create loss function
# Input 2 channels to the generator, (image and mask from 1st network)

# 