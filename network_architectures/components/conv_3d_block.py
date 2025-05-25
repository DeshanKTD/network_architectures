import torch
import torch.nn as nn

class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, 
                no_of_layers=2,
                kernel_size=3, 
                stride=1, 
                padding=1, 
                activation='relu',
                use_batch_norm=True,
                dropout=0.0
                ):
        super(ConvBlock3D, self).__init__()
        
        layers = []
        
        for i in range(no_of_layers):
            # Determine input and output channels for each layer
            conv_in_channels = in_channels if i == 0 else out_channels
            conv_out_channels = out_channels

            # Conv -> BatchNorm (optional) -> Activation
            layers.append(nn.Conv3d(conv_in_channels, conv_out_channels, kernel_size=kernel_size,
                                    stride=stride, padding=padding))
            if use_batch_norm:
                layers.append(nn.BatchNorm3d(conv_out_channels))
            layers.append(self._get_activation(activation))
        
        #   If the last layer is not followed by batch norm or dropout,
        if dropout > 0.0:
            layers.append(nn.Dropout3d(dropout))
            
        # create the sequential block
        self.block = nn.Sequential(*layers)
        
    def _get_activation(self,name):
            if name == 'relu':
                return nn.ReLU(inplace=True)
            elif name == 'leaky_relu':
                return nn.LeakyReLU(negative_slope=0.2, inplace=True)
            else:
                raise ValueError(f"Unsupported activation function: {name}")
            
    
    
    def get_activation(name):
        if name == 'relu':
            return nn.ReLU(inplace=True)
        elif name == 'leaky_relu':
            return nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            raise ValueError(f"Unsupported activation function: {name}")

    def forward(self, x):
        return self.block(x)