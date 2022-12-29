import torch
import glob
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A

from typing import List
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import CIFAR10
from albumentations.pytorch import ToTensorV2


class CifarDataset(CIFAR10):
    def __init__(self, root="~/", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)
        self.transform = transform

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


if __name__ == '__main__':

    train_transform = A.Compose([
        A.Resize(227, 227),
        A.HorizontalFlip(p=0.5),
        A.augmentations.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    train_dataloader = make_dataloder(transform=train_transform, train_=True, batch_size=32)

    img,label = next(iter(train_dataloader))

    print(img.shape, img[0])

    plt.imshow(img[0].permute(1,2,0))
    plt.show()