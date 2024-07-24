import torch.nn as nn

from atlas_ml.deeplearning.deeplearning import DeepLearning
from atlas_ml.deeplearning.utils import ResidualAdd


# implemented following the guidelines in section 4.1 in the paper https://arxiv.org/pdf/1711.00937
class Encoder(nn.Module):

    def __init__(self, in_channels:int, out_channels:int, latent_dim:int):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1),
            ResidualAdd(
                nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(256, 256, kernel_size=1),
                    nn.ReLU(),
                    nn.Conv2d(256, 256, kernel_size=1)
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(256, 256, kernel_size=1),
                    nn.ReLU(),
                    nn.Conv2d(256, 256, kernel_size=1)
                )
            )
        )

    
    def forward(self, x):
        output = self.model(x)
        return output



class Decoder(nn.Module):
    pass


class Quantizer(nn.Module):
    pass


class VQVAE(DeepLearning):
    
    def __init__(self, in_channels:int=1, out_channels:int=1, latent_dim:int=128) -> None:
        super(VQVAE, self).__init__()
        
        self.encoder = Encoder(in_channels=in_channels, out_channels=out_channels, latent_dim=latent_dim)
        self.decoder = Decoder()
        self.quantizer = Quantizer()


import torch

sample = torch.randn(2, 3, 128, 128)
encoder = Encoder(3, 3, 128)

output = encoder(sample)
print(output)