import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.preprocessing import MinMaxScaler
import warnings
from Lasa import ImportLASADataset, filename_list, file_list1, file_list2, file_list3, file_list4
from Dataset import FMDiffeoDataset
from torch.utils.data import Dataset, DataLoader
from utils import *
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# np.random.seed(42)

def inferece_plot(model,file_name,modelpath,visualpath):

    print("Trajectory:", file_name)

    # 准备数据
    positions, velocities, indices = ImportLASADataset(f'{file_name}.mat')
    x_min, x_max = positions[:, 0].min() - 20, positions[:, 0].max() + 20
    y_min, y_max = positions[:, 1].min() - 20, positions[:, 1].max() + 20
    limit=np.array([[x_min,y_min],[x_max,y_max]])
    target = np.array([0, 0])
    demonstrations={
        'positions': positions,
        'velocities': velocities,
        'target': target
    }
    normalized_demos, scale = prepare_demonstrations(demonstrations)
    positions, velocities, targets = create_training_data(normalized_demos)

    # 测试
    model.load_state_dict(torch.load(modelpath))
    scaled_limit = np.array(limit) / scale
    plot_potential_velocity_field(model, scaled_limit, positions, f'{visualpath}/p_v_field_{file_name}.png')


dim=2
h=64
l=5
from models import *
model = SDSEF_FM(dim=dim, hidden_dim=h, num_layers=l).to(device)
for file_name in file_list1:
    modelpath = f'models/SDSEF_FM_h{h}_l{l}/SDSEF_FM_{file_name}.pth'
    visualpath = f'visualizations/potential_velocity_field'
    inferece_plot(model,file_name,modelpath,visualpath)