import torch
import torch.nn as nn
from einops import rearrange

from atlas_ml.deeplearning.deeplearning import DeepLearning
from atlas_ml.deeplearning.utils import ResidualAdd


# implemented following the guidelines in section 4.1 in the paper https://arxiv.org/pdf/1711.00937
class Encoder(nn.Module):

    def __init__(self, in_channels:int, hidden_dim:int=256):
        super(Encoder, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=4, stride=2, padding=1),
            ResidualAdd(
                nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
                    nn.ReLU(),
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
                    nn.ReLU(),
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
                )
            )
        )

    
    def forward(self, x):
        output = self.model(x)
        return output



class Decoder(nn.Module):
    
    def __init__(self, input_channels:int=256, output_channels:int=3):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels=input_channels, out_channels=input_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=input_channels, out_channels=input_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=input_channels, out_channels=output_channels, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, z_q):
        x_reconstructed = self.model(z_q)
        return x_reconstructed

class Quantizer(nn.Module):
    
    def __init__(self, codebook_dim:int=256, latent_dim:int=128):
        super(Quantizer, self).__init__()
        self.codebook = nn.Embedding(codebook_dim, latent_dim) # codebook of KxD
        self.codebook.weight.data.uniform_(-1/codebook_dim, 1/codebook_dim) # some kind of normalisation


    def forward(self, z):
        b, _, h, w = z.shape
        # flatten for better handling of the input tensor
        z = rearrange(z, 'b d h w -> (b h w) d')
        # calculate the Euclidean distance --> ||z - e||^2 = ||z||^2 + ||e||^2 - 2 * z.e
        distances = (torch.sum(z**2, dim=1, keepdim=True)
                     + torch.sum(self.codebook.weight**2, dim=1)
                     - 2 * torch.matmul(z, self.codebook.weight.T)) # [B*H*W, D]
        
        # find nearest embedding for each latent vector in the codebook
        indices = torch.argmin(distances, dim=1).unsqueeze(1) # [B*H*W, 1]

        # convert indices to 1-hot encodings
        one_hot_encodings = torch.zeros(indices.shape[0], self.codebook.weight.shape[0], device=z.device)
        one_hot_encodings.scatter_(1, indices, 1)  # [B*H*W, K]

        # matmul between encodings and codebook to quantize the embeddings
        quantized = torch.matmul(one_hot_encodings, self.codebook.weight) # [B*H*W, D]

        # reshape to prepare dimensions for decoder
        quantized = rearrange(quantized, '(b h w) d -> b d h w', b=b, h=h, w=w)

        return quantized

class VQVAE(DeepLearning):
    
    def __init__(self, in_channels:int=1, hidden_dim:int=256, codebook_dim:int=512) -> None:
        super(VQVAE, self).__init__()
        
        self.encoder = Encoder(in_channels=in_channels, hidden_dim=hidden_dim)
        self.quantizer = Quantizer(codebook_dim=codebook_dim, latent_dim=hidden_dim)
        self.decoder = Decoder(input_channels=hidden_dim, output_channels=in_channels)

    def forward(self, x: torch.tensor) -> torch.tensor:
        output = self.encoder(x)
        quantized_output = self.quantizer(output)
        x_reconstructed = self.decoder(quantized_output)
        return x_reconstructed

    def train_model(self, **kwargs) -> None:
        return super().train_model(**kwargs)
    
    def _train_one_epoch(self, *kwargs) -> float:
        return super()._train_one_epoch(*kwargs)

    def _val_one_epoch(self, *kwargs) -> float:
        return super()._val_one_epoch(*kwargs)

import torch

sample = torch.randn(2, 3, 128, 128)
vq_vae = VQVAE(in_channels=3, hidden_dim=256, codebook_dim=512)

output = vq_vae(sample)
print(output)