import torch
from torch import nn
from model import Model
import torch.optim as optim
from dataset import *
import os
from config import *
from common import *

model = load_model(True)
model2 = load_model2(True)
optimizer = optim.Adam(model.parameters(), 1e-3)
optimizer2 = optim.Adam(model2.parameters(), 1e-3)
lossFn = nn.MSELoss().to(DEVICE)
for i in range(1):
    for data in trainDataloader:
        x, y = data
        x: torch.Tensor = x.to(DEVICE)
        y: torch.Tensor = y.to(DEVICE)
        prev = model(x)
        loss1 = lossFn(prev, y)
        prev2 = model2(prev)
        loss2 = lossFn(prev2, x)
        loss = loss1 + loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prev3 = model2(y)
        loss3 = lossFn(prev3, x)
        optimizer2.zero_grad()
        loss3.backward()
        optimizer2.step()

        print(f"gloss:{loss.item()} dloss:{loss3.item()}")

        torch.save(model.state_dict(), "model.pt")
