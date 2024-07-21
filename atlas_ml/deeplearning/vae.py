import torch
import torch.nn as nn
from atlas_ml.deeplearning.deeplearning import DeepLearning
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import cv2

from atlas_ml.transforms import TrimPixels
import math

class Encoder(nn.Module):

    def __init__(
            self, 
            input_channels:int=3, 
            hidden_dims:list=[32, 64, 128, 256, 512], 
            img_size:int=28,
            latent_dim:int=20
            ):
        super(Encoder, self).__init__()        
        encoder_modules = []
        for h in hidden_dims:
            encoder_modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=input_channels, out_channels=h, kernel_size=3, stride=2, padding=1),
                    nn.ReLU()
                )
            )
            input_channels = h
            conv_shape = math.floor((img_size+2*1-1*(3-1)-1)/2)+1
            img_size = conv_shape
        

        self.encoder = nn.Sequential(*encoder_modules)

        self.fc_mu = nn.Linear(hidden_dims[-1]*conv_shape*conv_shape, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1]*conv_shape*conv_shape, latent_dim)

    def forward(self, x):

        encoded_x = self.encoder(x)

        encoded_x = torch.flatten(encoded_x, start_dim=1)
        mu = self.fc_mu(encoded_x)
        logvar = self.fc_logvar(encoded_x)

        return mu, logvar
    

class Decoder(nn.Module):

    def __init__(
            self, 
            latent_dim:int=20, 
            output_channels:int=3, 
            hidden_dims:list=[512, 256, 128, 64, 32], 
            img_size:int=28,
            ):
        super(Decoder, self).__init__()
        self.hidden_dims = hidden_dims

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.ReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels=output_channels,
                                      kernel_size=3, padding=1),
                            TrimPixels(img_size),
                            nn.Sigmoid())

        decoder_modules = []
        self.conv_shape = math.floor((img_size+2*1-1*(3-1)-1)/2)+1
        img_size = self.conv_shape
        for idx in range(len(hidden_dims)-1):
            decoder_modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=hidden_dims[idx], 
                        out_channels=hidden_dims[idx+1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1),
                    nn.ReLU()
                )
            )
            self.conv_shape = math.floor((img_size+2*1-1*(3-1)-1)/2)+1
            img_size = self.conv_shape

        self.fc = nn.Linear(latent_dim, hidden_dims[0]*self.conv_shape*self.conv_shape)

        self.decoder = nn.Sequential(*decoder_modules)


    def forward(self, z):

        x = self.fc(z)

        x = x.view(-1, self.hidden_dims[0], self.conv_shape, self.conv_shape)

        x_reconstructed = self.final_layer(self.decoder(x))

        return x_reconstructed


class VAE(DeepLearning):

    def __init__(self, 
            input_channels:int=3, 
            hidden_dims:list=[32, 64, 128, 256, 512], 
            latent_dim:int=20,
            output_channels:int=3,
            img_size:int=28
            ):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_channels=input_channels, hidden_dims=hidden_dims, latent_dim=latent_dim, img_size=img_size)
        self.decoder = Decoder(latent_dim=latent_dim, output_channels=output_channels, hidden_dims=hidden_dims[::-1], img_size=img_size)


    def reparametrization(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        z = mu + std * eps

        return z
    

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
            criterion = nn.MSELoss(reduction='none')
        else:
            criterion = kwargs['criterion'](reduction='none')

        # get epochs
        assert 'epochs' in kwargs, "Please specify the number of epochs to start training"
        n_epochs = kwargs['epochs']
        start_epochs = 0 # TODO: for now is set to 0, if we load checkpoints this needs to be updated accordingly

        train_loop = tqdm(range(start_epochs, n_epochs), total=n_epochs-start_epochs)

        # get device
        device = kwargs['device']

        self.to(device)

        for e in train_loop:
            train_loss = self._train_one_epoch(
                optimizer=optimizer,
                train_dl=train_dl,
                criterion=criterion,
                device=device
            )
            val_loss = self._val_one_epoch(val_dl=val_dl, criterion=criterion, device=device, e=e)

            # TODO: set checkpoint here!
            
            train_loop.set_description(f'Train Loss: {round(train_loss, 2)}, Val Loss: {round(val_loss, 2)}')


    def _train_one_epoch(self, optimizer:Optimizer, train_dl:DataLoader, criterion, **kwargs) -> float:
        device = kwargs['device']
        self.train()
        train_loss = 0.0
        for data in train_dl:
            imgs = data[0].to(device)
            batch_size=data[0].shape[0]
            
            imgs_reconstructed, z, mu, logvar = self.forward(imgs)
            
            kl_div = torch.mean(-0.5*torch.sum(1+logvar
                                - mu**2
                                - torch.exp(logvar),
                                axis=1), dim=0)

            pixelwise_loss = criterion(imgs_reconstructed, imgs).view(batch_size, -1).sum(axis=1) # sum pixels loss per batch
            pixelwise_loss = pixelwise_loss.mean() # average over batch dimension

            loss = pixelwise_loss + kl_div

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
            batch_size = data[0].shape[0]

            imgs_reconstructed, z, mu, logvar = self.forward(imgs)

            kl_div = torch.mean(-0.5*torch.sum(1+logvar
                                                - mu**2
                                                - torch.exp(logvar),
                                                axis=1), dim=0)

            pixelwise_loss = criterion(imgs_reconstructed, imgs).view(batch_size, -1).sum(axis=1) # sum pixels loss per batch
            pixelwise_loss = pixelwise_loss.mean() # average over batch dimension

            loss = pixelwise_loss + kl_div
            val_loss += loss.item()

        # visually checking the images
        if kwargs['e'] % 5 == 0:
            cv2.imwrite(f'./sample_{kwargs['e']}.png', torch.permute(imgs_reconstructed[0]*255, (1, 2, 0)).cpu().detach().numpy())

        
        return val_loss/len(val_dl)



    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparametrization(mu, logvar)
        x_reconstructed = self.decoder(z)

        return x_reconstructed, z, mu, logvar
