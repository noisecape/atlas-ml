import torch.nn as nn


class TrimPixels(nn.Module):

    def __init__(self, img_size:int=28):
        super(TrimPixels, self).__init__()
        self.img_size=img_size

    def forward(self, x):
        return x[:, :, :self.img_size, :self.img_size]