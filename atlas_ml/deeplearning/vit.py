# implementation based on https://github.com/FrancescoSaverioZuppichini/ViT

import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F

class PatchEmbedding(nn.Module):

    def __init__(self, in_channels:int=3, emb_size:int=512, patch_size:int=16, img_size:int=224):
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
        x = torch.cat([cls_token, x], dim=1) # concat on the 1st dim
        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, emb_size:int=512, n_heads:int=8, dropout:float=0):
        super(MultiHeadAttention, self).__init__()

        self.emb_size=emb_size
        self.n_heads=n_heads
        self.dropout=dropout
        assert emb_size % n_heads == 0, "emb_size must be divisible by n_heads"
        self.qkv = nn.Linear(emb_size, emb_size*3) # fuse qkv matrices into 1 matrix, faster!
        self.att_dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
    
    def forward(self, x, mask=None):
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.n_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        scaling_factor = self.emb_size **0.5
        energies = torch.einsum("bhqd,bhkd -> bhqk", queries, keys)/scaling_factor
        if mask:
            energies = energies.mask_fill(mask==0, float("-1e20"))
        
        attention = F.softmax(energies, dim=-1)
        attention = self.att_dropout(attention)

        out = torch.einsum("bhal, bhlv -> bhav", attention, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super(ResidualAdd, self).__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class MLP(nn.Sequential):
    def __init__(self, emb_size:int, expansion:int=4, drop_p:float=0.):
        super(MLP, self).__init__(
            nn.Linear(emb_size, emb_size*expansion),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(emb_size*expansion, emb_size)
        )


class TransformerBlock(nn.Sequential):
    def __init__(self,
                 emb_size:int=512,
                 drop_p=0.3,
                 forward_expansion:int=4,
                 forward_drop_p:float=0.1,
                 **kwargs):
        super(TransformerBlock, self).__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size=emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MLP(
                    emb_size=emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                    nn.Dropout(drop_p)
                )
            ))

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth:int=12, **kwargs):
        super(TransformerEncoder, self).__init__(*[TransformerBlock(**kwargs) for _ in range(depth)])

class ViT(nn.Sequential):
    def __init__(self,
                 in_channels:int=3,
                 patch_size:int=16,
                 emb_size:int=512,
                 img_size:int=224,
                 depth:int=12,
                 **kwargs):
        super(ViT, self).__init__(
            PatchEmbedding(in_channels=in_channels, emb_size=emb_size, patch_size=patch_size, img_size=img_size),
            TransformerEncoder(depth=depth, emb_size=emb_size, **kwargs)
        )

# sample = torch.randn(6, 3, 224, 224)
# from torchsummary import summary

# summary(ViT().to('cuda'), (3, 224, 224), batch_size=256, device='cuda')