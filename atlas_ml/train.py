from datasets.data_pipeline import DataPipeline
from deeplearning.vae import VAE
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import cv2
from PIL import Image

def train():

    data_pipeline = DataPipeline()

    mnist_train_dl = data_pipeline.get_dataset('mnist')

    model = VAE(input_channels=1, latent_dim=20, output_channels=1, img_size=28).to('cuda')

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    reconstruction_criterion = nn.MSELoss(reduction='none')

    for e in range(50):
        train_loop = tqdm(mnist_train_dl, total=len(mnist_train_dl))
        model.train()
        for data in train_loop:
            imgs = data[0].to('cuda')
            batch_size = data[0].shape[0]
            _ = data[1]
            
            imgs_reconstructed, z, mu, logvar = model(imgs)

            kl_div = torch.mean(-0.5 * torch.sum(1 + logvar 
                                      - mu**2 
                                      - torch.exp(logvar), 
                                      axis=1), dim=0)
    
            pixelwise_loss = reconstruction_criterion(imgs_reconstructed, imgs).view(batch_size, -1).sum(axis=1) # sum losses of pixels per img in the batch
            pixelwise_loss = pixelwise_loss.mean() # average over batch dimension
            
            loss = pixelwise_loss + kl_div
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loop.set_description(f'Loss: {round(loss.item(), 2)}')

            if e % 5 == 0:

                cv2.imwrite(f'./sample_{e}.png', torch.permute(imgs_reconstructed[0]*255, (1, 2, 0)).cpu().detach().numpy())




if __name__ == "__main__":
    train()