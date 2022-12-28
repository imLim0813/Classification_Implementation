import torch
import glob
import numpy as np
import albumentations as A

from typing import List
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import CIFAR10


def make_dataloder(transform: A.Compose, train_: bool, batch_size: int):
    """
    CIFAR 10 데이터 셋을 반환하는 함수
    """

    return DataLoader(dataset=CIFAR10(root='~/', transform=transform, train=train_),
                      batch_size=batch_size, shuffle=train_, drop_last=True)