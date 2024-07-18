import torch.nn as nn
from atlas_ml.types_ import *
from abc import abstractmethod, ABC

class DeepLearning(ABC, nn.Module):

    @abstractmethod
    def forward(self, *x:Tensor) -> Tensor:
        pass

    @abstractmethod
    def train_model(self, **kwargs) -> None:
        pass

    @abstractmethod
    def _train_one_epoch(self, *kwargs) -> float:
        pass

    @abstractmethod
    def _val_one_epoch(self, *kwargs) -> float:
        pass
    