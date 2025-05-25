from network_architectures.networks.auto_encoder.vae.vae_decoder import VAEDecoder
from network_architectures.networks.auto_encoder.vae.vae_encoder import VAEEncoder3D
import torch
import torch.nn as nn

class VAE3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, latent_dims=256,feature_channels=[32, 64, 128],input_shape=(256, 256, 256)):
        super(VAE3D, self).__init__()
        self.encoder = VAEEncoder3D(in_channels, feature_channels=feature_channels, latent_dims=latent_dims, input_shape=input_shape)
        # The initial shape is determined by the input size and the feature channels
        
        initial_shape = (
            feature_channels[-1], 
            input_shape[0] // (2 ** len(feature_channels)),
            input_shape[1] // (2 ** len(feature_channels)),
            input_shape[2] // (2 ** len(feature_channels))
            )  # Assuming input size is reduced to 128x4x4x4
        self.decoder = VAEDecoder(feature_channels=feature_channels[::-1], latent_dims=latent_dims, initial_shape=initial_shape)
        self.final_conv = nn.Conv3d(in_channels=feature_channels[0], out_channels=out_channels, kernel_size=1)
        
    def vae_loss(recon_x, x, mu, logvar):
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_div
            
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.encoder.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        print(f"x_recon shape: {x_recon.shape}")
        x_recon = self.final_conv(x_recon)
        return x_recon, mu, logvar
        # return mu, logvar
        
    
