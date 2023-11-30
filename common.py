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
