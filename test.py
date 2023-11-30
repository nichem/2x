from model import *
from config import *
import os
from common import *

model = load_model(False)
from dataset import *

input, target = testDataset[0]
input = input.to(DEVICE)
target = target.to(DEVICE)
output = model(torch.unsqueeze(input, 0))

utils.save_image(input, "out/test_input.jpg")
utils.save_image(output, "out/test_output.jpg")
utils.save_image(target, "out/test_target.jpg")
