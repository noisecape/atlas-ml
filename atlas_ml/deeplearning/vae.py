import torch
import torch.nn as nn

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


class VAE(nn.Module):

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


    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparametrization(mu, logvar)
        x_reconstructed = self.decoder(z)

        return x_reconstructed
    


# x = torch.randn(10, 3, 224, 224)
# vae = VAE(img_size=224)

# x_reconstructed = vae(x)

# print(x.shape, x_reconstructed.shape)