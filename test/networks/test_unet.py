import torch
from network_architectures.networks.unet.attention.bam.bam_unet_3d import BAMUNet3D
from network_architectures.networks.unet.attention.cbam.cbam_unet_3d import CBAMUNet3D


def test_bam_unet_3d_output_shape():
    # Initialize the BAMUNet3D model
    model = BAMUNet3D(in_channels=1, out_channels=1, feature_channels=[8, 16, 32, 64])
    
    # Create a random input tensor with shape (batch_size, channels, depth, height, width)
    x = torch.randn(1, 1, 128, 128, 128)
    
    # Forward pass through the model
    output = model(x)
    
    # Check the output shape
    assert output.shape == (1, 1, 128, 128, 128), f"Expected output shape (1, 1, 128, 128, 128), but got {output.shape}"



def test_cbam_unet_3d_output_shape():
    # Initialize the BAMUNet3D model
    model = CBAMUNet3D(in_channels=1, out_channels=1, feature_channels=[8, 16, 32, 64])
    
    # Create a random input tensor with shape (batch_size, channels, depth, height, width)
    x = torch.randn(1, 1, 128, 128, 128)
    
    # Forward pass through the model
    output = model(x)
    
    # Check the output shape
    assert output.shape == (1, 1, 128, 128, 128), f"Expected output shape (1, 1, 128, 128, 128), but got {output.shape}"
