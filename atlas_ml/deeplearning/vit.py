# implementation based on https://github.com/FrancescoSaverioZuppichini/ViT

import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange


class PatchEmbedding(nn.Module):

    def __init__(self, in_channels:int=3, emb_size:int=384, patch_size:int=16, img_size:int=224):
        super(PatchEmbedding, self).__init__()
        self.patch_size=patch_size
        self.projection=nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e')
        )
        self.cls_token=nn.Parameter(torch.randn(1,1,emb_size))
        self.positions = nn.Parameter(torch.randn((img_size//patch_size) **2 + 1, emb_size))

    
    def forward(self, x):
        b, _, _, _ = x.shape
        x=self.projection(x)
        cls_token = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_token, x], dim=1)
        return x
    

sample = torch.randn(6, 3, 224, 224)
pathembs = PatchEmbedding()
embeds = pathembs(sample)
print(embeds.shape)