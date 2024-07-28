import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from atlas_ml.deeplearning.deeplearning import DeepLearning
from atlas_ml.deeplearning.utils import ResidualAdd
from atlas_ml.transforms import TrimPixels
import cv2


# implemented following the guidelines in section 4.1 in the paper https://arxiv.org/pdf/1711.00937
class Encoder(nn.Module):

    def __init__(self, input_channels:int, hidden_dims:List[int]=[32, 64], img_size:int=28, latent_dim:int=128):
        super(Encoder, self).__init__()
        
        encoder_layer = []
        for l in hidden_dims:
            encoder_layer.append(
            nn.Sequential(
                nn.Conv2d(in_channels=input_channels, out_channels=l, kernel_size=4, stride=2, padding=1),
                nn.GELU()
            )
        )
            input_channels = l
            conv_shape = math.floor((img_size+2*1-1*(3-1)-1)/2)+1
            img_size = conv_shape
            
        self.downsample = nn.Sequential(*encoder_layer)

        self.residual = nn.Sequential(
            ResidualAdd(
                nn.Sequential(
                    nn.GELU(),
                    nn.Conv2d(hidden_dims[-1], hidden_dims[-1], kernel_size=1),
                    nn.GELU(),
                    nn.Conv2d(hidden_dims[-1], hidden_dims[-1], kernel_size=1)
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    nn.GELU(),
                    nn.Conv2d(hidden_dims[-1], hidden_dims[-1], kernel_size=1),
                    nn.GELU(),
                    nn.Conv2d(hidden_dims[-1], hidden_dims[-1], kernel_size=1)
                )
            )
        )

        self.pre_quantization = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(in_channels=hidden_dims[-1], out_channels=latent_dim, kernel_size=1)
        )

    def forward(self, x):
        output = self.downsample(x) # downsample
        output = self.residual(output) # residual layers
        output = self.pre_quantization(output) # 1x1 conv to match codebook dim
        return output

class Decoder(nn.Module):
    
    def __init__(self, in_channels:int=128, output_channels:int=3, upsample:bool=False, scale_factor:int=2, expansion_factor:int=4, img_size:int=28):
        super(Decoder, self).__init__()
        upsample_layers = []

        if upsample:
            # use combinatino of upsample+conv2d to avoid checkboard-artefacts
            for _ in range(expansion_factor-1):
                upsample_layers.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=scale_factor),
                        nn.Conv2d(in_channels=in_channels, out_channels=in_channels//scale_factor, kernel_size=1, padding=1),
                        nn.GELU()
                    )
                )
                in_channels = in_channels //scale_factor

            upsample_layers.append(nn.Sequential(
                nn.Upsample(scale_factor=scale_factor),
                nn.Conv2d(in_channels=in_channels, out_channels=output_channels, kernel_size=1),
                TrimPixels(img_size=img_size)
            ))
        else:
            # standard conv transposed. it might introduce checkboard-artefacts
            for _ in range(expansion_factor-1):
                upsample_layers.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels//scale_factor, kernel_size=4, stride=2, padding=1, output_padding=1),
                        nn.GELU()
                    )
                )
                in_channels = in_channels // scale_factor

            upsample_layers.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=output_channels, kernel_size=4, stride=2, padding=1, output_padding=1),
                TrimPixels(img_size=img_size),
                nn.Sigmoid()
            ))
        self.decoder = nn.Sequential(*upsample_layers)

    def forward(self, z_q):
        x_reconstructed = self.decoder(z_q)
        return x_reconstructed

class Quantizer(nn.Module):
    
    def __init__(self, codebook_dim:int=256, latent_dim:int=128, beta:float=0.2):
        super(Quantizer, self).__init__()
        self.codebook = nn.Embedding(codebook_dim, latent_dim) # codebook of KxD
        self.codebook.weight.data.uniform_(-1/codebook_dim, 1/codebook_dim) # some kind of normalisation
        self.beta = beta

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
        
        codebook_loss = F.mse_loss(z.detach(), quantized) # move the codebook's latent towards the encoder's output.
        commitment_loss = self.beta * F.mse_loss(z, quantized.detach()) # commits the encoder to output vectors close to the codebook's
        
        # Make the gradient with respect to latents be equal to the gradient with respect to quantized latents 
        quantized = z + (quantized - z).detach()

        # reshape to prepare dimensions for decoder
        quantized = rearrange(quantized, '(b h w) d -> b d h w', b=b, h=h, w=w)

        return quantized, codebook_loss, commitment_loss

class VQVAE(DeepLearning):
    
    def __init__(self, input_channels:int=1, output_channels:int=1, hidden_dims:list=[32, 64, 128, 256], codebook_dim:int=512, img_size:int=128, latent_dim:int=128, scale_factor:int=2, expansion_factor:int=2) -> None:
        super(VQVAE, self).__init__()
        self.encoder = Encoder(input_channels=input_channels, hidden_dims=hidden_dims, img_size=img_size, latent_dim=latent_dim)
        self.quantizer = Quantizer(codebook_dim=codebook_dim, latent_dim=latent_dim)
        self.decoder = Decoder(in_channels=latent_dim, output_channels=output_channels, scale_factor=scale_factor, expansion_factor=expansion_factor, img_size=img_size)

    def forward(self, x: torch.tensor) -> torch.tensor:
        output = self.encoder(x)
        quantized_output, codebook_loss, commitment_loss = self.quantizer(output)
        x_hat = self.decoder(quantized_output)
        return x_hat, codebook_loss, commitment_loss

    def train_model(self, **kwargs) -> None:
        # get dataloaders
        assert 'train_dataloader' in kwargs, 'Please specify a train dataloader to start training!'
        train_dl = kwargs['train_dataloader']
        assert 'val_dataloader' in kwargs, 'Please specify a val dataloader to start training!'
        val_dl = kwargs['val_dataloader']

        # get optimizer
        if 'optimizer' not in kwargs:
            lr = kwargs['lr'] if 'lr' in kwargs else 3e-4 # default lr if not specified
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        else:
            lr = kwargs['lr'] if 'lr' in kwargs else 3e-4 # default lr if not specified
            optimizer = kwargs['optimizer'](self.parameters(), lr=lr)

        # get criterion
        if 'criterion' not in kwargs:
            criterion = nn.MSELoss()
        else:
            criterion = kwargs['criterion']()

        # get epochs
        assert 'epochs' in kwargs, "Please specify the number of epochs to start training"
        n_epochs = kwargs['epochs']
        start_epochs = 0 # TODO: for now is set to 0, if we load checkpoints this needs to be updated accordingly

        train_loop = tqdm(range(start_epochs, n_epochs), total=n_epochs-start_epochs)

        # get device
        device = kwargs['device']

        self.to(device)

        start_epochs = 0

        for e in train_loop:
            train_loss = self._train_one_epoch(
                optimizer=optimizer,
                train_dl=train_dl,
                criterion=criterion,
                device=device
            )
            val_loss = self._val_one_epoch(
                val_dl=val_dl, 
                criterion=criterion, 
                device=device, 
                e=e
                )

            # TODO: set checkpoint here!
            
            train_loop.set_description(f'Train Loss: {round(train_loss, 2)}, Val Loss: {round(val_loss, 2)}')
    
    def _train_one_epoch(self, optimizer:Optimizer, train_dl:DataLoader, criterion, **kwargs) -> float:
        device = kwargs['device']
        self.train()
        train_loss = 0.0
        for data in train_dl:
            imgs = data[0].to(device)
            
            x_hat, codebook_loss, commitment_loss = self.forward(imgs)

            pixelwise_loss = criterion(x_hat, imgs) # average over batch dimension

            loss = pixelwise_loss + codebook_loss + commitment_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        return train_loss / len(train_dl)
        

    def _val_one_epoch(self, val_dl:DataLoader, criterion, **kwargs) -> float:
        device = kwargs['device']
        self.eval()
        val_loss = 0.0
        for data in val_dl:
            imgs = data[0].to(device)
            x_hat, codebook_loss, commitment_loss = self.forward(imgs)
            pixelwise_loss = criterion(x_hat, imgs)

            loss = pixelwise_loss + codebook_loss + commitment_loss

            val_loss += loss.item()
        
                # visually checking the images
        if kwargs['e'] % 5 == 0:
            cv2.imwrite(f'./sample_{kwargs['e']}.png', torch.permute(x_hat[0]*255, (1, 2, 0)).cpu().detach().numpy())

        
        return val_loss / len(val_dl)