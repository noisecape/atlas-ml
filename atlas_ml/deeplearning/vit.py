# implementation based on https://github.com/FrancescoSaverioZuppichini/ViT

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.data
from deeplearning import DeepLearning
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from atlas_ml.datasets.data_pipeline import DataPipeline
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.optim import Optimizer

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

class ViT(DeepLearning):
    def __init__(self,
                 in_channels:int=3,
                 patch_size:int=16,
                 emb_size:int=512,
                 img_size:int=224,
                 depth:int=12,
                 n_classes=10,
                 **kwargs):
        super(ViT, self).__init__()

        self.patch_embedding = PatchEmbedding(in_channels=in_channels, emb_size=emb_size, patch_size=patch_size, img_size=img_size)
        self.transformer_encoder = TransformerEncoder(depth=depth, emb_size=emb_size, **kwargs)
        self.mlp = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size), 
            nn.Linear(emb_size, n_classes)
        )
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        patch_embeddings = self.patch_embedding(x)
        output = self.transformer_encoder(patch_embeddings)
        output = self.mlp(output)
        return output
    
    def _train_one_epoch(self, optimizer:Optimizer, train_dl:DataLoader, criterion, **kwargs) -> float:
        device = kwargs['device']
        self.train()
        train_loss = 0.0
        for data in train_dl:
            imgs = data[0].to(device)
            labels = data[1].to(device)
            preds = self.forward(imgs)
            loss = criterion(preds, labels)

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
            labels = data[1].to(device)
            preds = self.forward(imgs)
            loss = criterion(preds, labels)

            val_loss += loss.item()
        
        return val_loss / len(val_dl)


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
            criterion = nn.CrossEntropyLoss()
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

        for e in train_loop:
            train_loss = self._train_one_epoch(
                optimizer=optimizer,
                train_dl=train_dl, 
                criterion=criterion, 
                device=device
                )
            val_loss = self._val_one_epoch(val_dl=val_dl, criterion=criterion, device=device)
            
            #TODO: save checkpoints

            print(train_loss, val_loss)


# sample = torch.randn(6, 3, 224, 224)
if __name__ == '__main__':
    dataset_config = {'batch_size':64, 'val_split':0.2, 'num_workers':2, 'pin_memory':True}
    vit = ViT(in_channels=1)
    train_dl, val_dl = DataPipeline(configs=dataset_config).get_dataset() # default is mnist

    vit.train_model(
        train_dataloader=train_dl, 
        val_dataloader=val_dl, 
        optimizer=torch.optim.Adam,
        criterion=nn.CrossEntropyLoss,
        epochs=30,
        lr=3e-4,
        device='cuda')
# from torchsummary import summary

# summary(ViT().to('cuda'), (3, 224, 224), batch_size=256, device='cuda')