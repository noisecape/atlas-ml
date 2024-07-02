import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):

    def __init__(self, input_channels:int=3, expanded_dim:int=32, hidden_dim:int=256, latent_dim:int=20):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=expanded_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(expanded_dim, expanded_dim*2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(expanded_dim*2, expanded_dim*4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        self.mlp = nn.Sequential(nn.Linear(expanded_dim*64, hidden_dim), nn.ReLU(inplace=True))
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):

        encoded_x = self.encoder(x)

        encoded_x = encoded_x.view(-1)
        z = self.mlp(encoded_x)
        mu = self.fc_mu(z)
        logvar = self.fc_logvar(z)

        return mu, logvar
    

class Decoder(nn.Module):

    def __init__(self, dim_latent_variable:int=20, expanded_dim:int=32, out_channels:int=3):
        super(Decoder, self).__init__()

        self.fc = nn.Linear(dim_latent_variable, expanded_dim*4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(expanded_dim*4, expanded_dim*2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(expanded_dim*2, expanded_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(expanded_dim, out_channels, kernel_size=3, stride=2, padding=1)
        )


    def forward(self, x):

        x = self.fc(x)

        x_reconstructed = F.sigmoid(self.decoder(x))

        return x_reconstructed


encoder = Encoder()
decoder = Decoder()
import torch
x = torch.randn(1, 3, 28, 28)
mu, logvar = encoder(x)

# simulating reparametrization
z = torch.randn(1, 20)
print(z.shape, mu.shape, logvar.shape)
x_reconstructed = decoder(z)
print(z.shape)