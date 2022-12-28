import torch
import glob
import numpy as np
import albumentations as A

from typing import List
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import CIFAR10


class CifarDataset(CIFAR10):
    def __init__(self, root="~/", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label


def make_dataloder(transform: A.Compose, train_: bool, batch_size: int):
    """
    CIFAR 10 데이터 셋을 반환하는 함수
    """

    return DataLoader(dataset=CifarDataset(root='~/', train=train_, download=True, transform=transform),
                      batch_size=batch_size, shuffle=train_, drop_last=True)