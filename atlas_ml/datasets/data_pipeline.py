from sklearn.datasets import load_diabetes
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class DataPipeline:

    def __init__(self, configs:dict=None):
        
        self.configs = configs

    def get_dataset(self, dataset_name:str):

        if dataset_name == 'diabetes':
            return load_diabetes()
        elif dataset_name == 'mnist':
            transform = transforms.ToTensor()
            dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
            dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
            return dataloader