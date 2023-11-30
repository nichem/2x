from torchvision import transforms, utils
from PIL import Image
import torch
import random, os
from torch.utils.data import Dataset, DataLoader
from config import *


class CustomDataset(Dataset):
    def __init__(self, imgSize: int, count: int):
        self.files = [os.path.join("data", i) for i in os.listdir("data")]
        self.imgSize = imgSize
        self.count = count
        self.t1 = transforms.Compose(
            [
                transforms.RandomCrop(imgSize),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
            ]
        )
        self.t2 = transforms.Resize(imgSize // 2, antialias=False)
        pass

    def __getitem__(self, index):
        file = random.choice(self.files)
        img = Image.open(file)
        if img.mode == "RGBA":
            img = img.convert("RGB")
        tensor = self.t1(img)
        tensor2 = self.t2(tensor)
        return tensor2, tensor

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return self.count


trainDataset = CustomDataset(IMAGE_SIZE * 2, 256)
trainDataloader = DataLoader(trainDataset, 64, drop_last=True)
testDataset = CustomDataset(IMAGE_SIZE * 2, 256)
testDataloader = DataLoader(testDataset, 64, drop_last=True)

if __name__ == "__main__":
    x, y = trainDataset[0]
    utils.save_image(x, "out/dataset_x.jpg")
    utils.save_image(y, "out/dataset_y.jpg")
    print(x.shape, y.shape)
