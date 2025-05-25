from network_architectures.components.conv_3d_block import ConvBlock3D
import torch

def test_conv_3d_block_output_shape():
    block = ConvBlock3D(in_channels=1,out_channels=4)
    x = torch.randn(1, 1, 128, 128, 128)  # Example input tensor with shape (batch_size, channels, depth, height, width)
    output = block(x)
    out = torch.randn(1, 4, 128, 128, 128)  # Example output tensor with shape (batch_size, channels, depth, height, width)
    
    assert output.shape == out.shape, f"Expected output shape {out.shape}, but got {output.shape}"