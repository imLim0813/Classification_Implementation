import torch
import albumentations as A
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from albumentations.pytorch import ToTensorV2


class CifarDataset(CIFAR10):
    """
    Description
        : A class to download "CIFAR10" dataset.
    """
    def __init__(self, root='~/', transform=None, train=True, download=True):
        super().__init__(root=root, train=train, download=download, transform=transform)
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            data = self.transform(image=image)
            image = data['image']
        else:
            mean=(0.485, 0.456, 0.406)
            std=(0.229, 0.224, 0.225)
            transform = A.Compose([A.Resize(224, 224), A.Normalize(mean, std), ToTensorV2()])
            
            data = transform(image=image)
            image = data['image']
        
        return image, label


def make_dataloder(transform: A.Compose, train_: bool, batch_size: int):
    """
    Description
        : A class to make "CIFAR10" dataloader.
    """

    return DataLoader(dataset=CifarDataset(root='~/', train=train_, download=True, transform=transform),
                      batch_size=batch_size, shuffle=train_, drop_last=True)

# Debug
if __name__ == '__main__':
    mean=(0.485, 0.456, 0.406)
    std=(0.229, 0.224, 0.225)

    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.augmentations.transforms.Normalize(mean, std),
        ToTensorV2()
    ])

    train_dataloader = make_dataloder(transform=train_transform, train_=True, batch_size=32)

    img,label = next(iter(train_dataloader))

    print(img.shape)

    orig_img = torch.zeros_like(img[0])

    # Normalize 된 텐서를 되돌리는 연산
    for idx, (mean_, std_) in enumerate(zip(mean, std)):
        orig_img[idx] = (img[0][idx] * std_ + mean_)

    plt.imshow(orig_img.permute(1,2,0))
    plt.show()