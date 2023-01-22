import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


def VGG_Block(in_channels, out_channels, repeat):
    vgg_block = []
    for i in range(repeat):
        vgg_block.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1))
        vgg_block.append(nn.ReLU())
        in_channels=out_channels
    vgg_block.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*vgg_block)


class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()

        self.conv_block_1 = VGG_Block(in_channels=3, out_channels=64, repeat=2)
        self.conv_block_2 = VGG_Block(in_channels=64, out_channels=128, repeat=2)
        self.conv_block_3 = VGG_Block(in_channels=128, out_channels=256, repeat=3)
        self.conv_block_4 = VGG_Block(in_channels=256, out_channels=512, repeat=3)
        self.conv_block_5 = VGG_Block(in_channels=512, out_channels=512, repeat=3)

        self.classifier = nn.Sequential(nn.Linear(in_features=512*7*7, out_features=4096),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(in_features=4096, out_features=1024),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(in_features=1024, out_features=10))

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        x = self.conv_block_5(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    model = VGGNet()
    summary(model, (1, 3, 224, 224))
