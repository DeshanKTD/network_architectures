import torch
from network_architectures.networks.auto_encoder.vae.variatinal_auto_encoder import VAE3D


def test_vae_3d_output_shape():
    # Initialize the BAMUNet3D model
    model = VAE3D(in_channels=1, out_channels=1, latent_dims=256, feature_channels=[16, 32, 64], input_shape=(256,256,256))
    
    # Create a random input tensor with shape (batch_size, channels, depth, height, width)
    x = torch.randn(1, 1, 256, 256, 256)
    
    # Forward pass through the model
    output,_,_ = model(x)
    
    # Check the output shape
    assert output.shape == (1, 1, 256, 256, 256), f"Expected output shape (1, 1, 256, 256, 256), but got {output.shape}"