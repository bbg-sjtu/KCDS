import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.preprocessing import MinMaxScaler
import warnings
from Lasa import Import3DDataset, file_list_3D_good
from Dataset import FMDiffeoDataset
from torch.utils.data import Dataset, DataLoader
from utils import *
import os
from models import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# np.random.seed(42)

def velocity_model_save(model,modelpath):
    velocity_model=model.diffeo.velocity_field
    torch.save(velocity_model.state_dict(), modelpath)
    print(f"Saved to {modelpath}")

if __name__ == "__main__":

    for file_name in file_list_3D_good:
        dim=3
        h=128
        l=3
        model = SDSEF_FM(dim=dim, hidden_dim=h, num_layers=l).to(device)
        model.load_state_dict(torch.load(f'models/SDSEF_FM_h{h}_l{l}_3D/SDSEF_FM_{file_name}.pth'))
        modelpath = f'models/velocity_models_3d/velocity_model_h{h}_l{l}_{file_name}.pth'
        velocity_model_save(model,modelpath)