import torch
from torch import nn
from torchvision import transforms, utils
from PIL import Image
from config import *


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        def block(i, o):
            return [
                nn.Conv2d(i, o, 3, 1, 1),
                nn.Conv2d(o, o, 3, 1, 1),
                # nn.BatchNorm2d(o),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ]

        self.cnn1 = nn.Sequential(*block(3, 64))
        self.cnn2 = nn.Sequential(*block(64, 128))

        def block2(i, o):
            return [
                nn.ConvTranspose2d(i, i, 3, 1, 1),
                # nn.BatchNorm2d(64),
                nn.ConvTranspose2d(i, o, 3, 2, 1, 1),
                # nn.BatchNorm2d(3),
                nn.ReLU(),
            ]

        self.transCnn1 = nn.Sequential(*block2(128, 64))
        self.transCnn2 = nn.Sequential(*block2(64, 64))
        self.transCnn3 = nn.Sequential(*block2(64, 3)[:-1], nn.Sigmoid())

    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.transCnn1(x)
        x = self.transCnn2(x)
        x = self.transCnn3(x)
        return x


class Model2(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        def block(i, o):
            return [
                nn.Conv2d(i, o, 3, 1, 1),
                nn.Conv2d(o, o, 3, 1, 1),
                # nn.BatchNorm2d(o),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ]

        self.cnn = nn.Sequential(*block(3, 64))
        self.cnn2 = nn.Sequential(
            nn.Conv2d(64, 3, 3, 1, 1), nn.Conv2d(3, 3, 3, 1, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.cnn2(x)

        return x


if __name__ == "__main__":
    model = Model()
    model2 = Model2()
    input = torch.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE))
    output = model(input)
    output2 = model2(output)
    print(output.shape, output2.shape)
    # print(model)
