from sklearn.datasets import load_diabetes
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class DataPipeline:

    def __init__(self, configs:dict=None):
        
        self.configs = configs

    def get_dataset(self, dataset_name:str='mnist'):

        if dataset_name == 'diabetes':
            return load_diabetes()
        elif dataset_name == 'mnist':
            transform = transforms.ToTensor()
            trainset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
            testset = datasets.MNIST(root='./data', train=False,transform=transform, download=True)
            # calculating the splits
            train_size = int((1-self.configs['val_split']) * len(trainset))
            val_size = len(trainset) - train_size

            trainset, valset = random_split(trainset, [train_size, val_size])
            train_load = DataLoader(trainset, batch_size=self.configs['batch_size'], shuffle=True, num_workers=self.configs['num_workers'], pin_memory=self.configs['pin_memory'])
            val_load = DataLoader(valset, batch_size=self.configs['batch_size'], shuffle=False, num_workers=self.configs['num_workers'], pin_memory=self.configs['pin_memory'])
            return train_load, val_load