from model import *
from config import *
import os


def load_model(train: bool) -> Model:
    model = Model().to(DEVICE)
    if os.path.exists("model.pt"):
        model.load_state_dict(torch.load("model.pt", map_location=DEVICE))
    if train:
        model.train()
    else:
        model.eval()
    return model

def load_model2(train: bool) -> Model2:
    model = Model2().to(DEVICE)
    if os.path.exists("model2.pt"):
        model.load_state_dict(torch.load("model2.pt", map_location=DEVICE))
    if train:
        model.train()
    else:
        model.eval()
    return model