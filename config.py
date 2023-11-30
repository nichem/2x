import torch

IMAGE_SIZE = 128
DEVICE = "cuda" if (torch.cuda.is_available()) else "cpu"
