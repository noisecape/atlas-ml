from datasets.data_pipeline import DataPipeline
from deeplearning.vae import VAE
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import cv2

def train():

    data_pipeline = DataPipeline()

    mnist_train_dl = data_pipeline.get_dataset('mnist')

    model = VAE(input_channels=1, latent_dim=20, output_channels=1, img_size=28).to('cuda')

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    reconstruction_criterion = nn.MSELoss()

    for e in range(50):
        train_loop = tqdm(mnist_train_dl, total=len(mnist_train_dl))
        for data in train_loop:
            imgs = data[0].to('cuda')
            _ = data[1]
            
            imgs_reconstructed, z, mu, logvar = model(imgs)

            reconstruction_loss = reconstruction_criterion(imgs_reconstructed, imgs)
            kl_divergence_loss = torch.mean(-0.5 *torch.sum(1+logvar - mu**2 - torch.exp(logvar), dim=1))

            loss = reconstruction_loss + kl_divergence_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loop.set_description(f'Loss: {round(loss.item(), 2)}')

            if e % 5 == 0:
                cv2.imwrite(f'./sample_{e}.png', torch.permute(imgs_reconstructed[0], (2, 1, 0)).cpu().detach().numpy())




if __name__ == "__main__":
    train()