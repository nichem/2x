import torch
from torch import nn
from model import Model
import torch.optim as optim
from dataset import *
import os
from config import *
from common import *

model = load_model(True)
optimizer = optim.Adam(model.parameters(), 1e-3)
lossFn = nn.MSELoss().to(DEVICE)
model.train()
for i in range(1):
    for data in trainDataloader:
        x, y = data
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        prev = model(x)
        loss = lossFn(prev, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"loss:{loss.item()}")
    torch.save(model.state_dict(), "model.pt")
