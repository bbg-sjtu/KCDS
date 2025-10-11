import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.preprocessing import MinMaxScaler
import warnings
from Lasa import ImportLASADataset, filename_list
from Dataset import FMDiffeoDataset
from torch.utils.data import Dataset, DataLoader
from utils import *
from models import *
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(42)

dim=2
h=64
l=8
lr=0.005
weight_decay=1e-8
num_epochs=50
model_path = f"models/SDSEF_FM_h{h}_l{l}"
visual_path = f"visualizations/SDSEF_FM_h{h}_l{l}"
os.makedirs(model_path, exist_ok=True)
os.makedirs(visual_path, exist_ok=True)

for file_name in ['BendedLinePy', 'Leaf_1Py', 'Multi_Models_3Py', 'Multi_Models_4Py', 'heeePy', 'Leaf_2Py', 'SnakePy']:
    print("Trajectory:", file_name)
    # 准备数据
    positions, velocities, indices = ImportLASADataset(f'{file_name}.mat')
    x_min, x_max = positions[:, 0].min() - 20, positions[:, 0].max() + 20
    y_min, y_max = positions[:, 1].min() - 20, positions[:, 1].max() + 20
    # limit=[x_min, x_max, y_min, y_max]
    limit=np.array([[x_min,y_min],[x_max,y_max]])
    target = np.array([0, 0])
    demonstrations={
        'positions': positions,
        'velocities': velocities,
        'target': target
    }
    normalized_demos, scale = prepare_demonstrations(demonstrations)
    positions, velocities, targets = create_training_data(normalized_demos)
    dataset = FMDiffeoDataset(positions, velocities, targets)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    # 初始化模型
    model = SDSEF_FM(dim=dim, hidden_dim=h, num_layers=l).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # 训练循环
    losses = []
    best_train_loss = float('inf')
    for epoch in range(num_epochs):
        total_loss=0
        for _, (x, v, targets) in enumerate(loader):

            optimizer.zero_grad()
            # 前向传递 - 使用直接速度预测以提高效率
            pred_velocities = model.direct_velocity_prediction(x, targets)
            # pred_velocities = model.forward(x, targets)

            # 计算损失
            loss = criterion(pred_velocities, v)

            # 反向传播
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            total_loss += loss.item()
        
        print(f'Epoch {epoch}, Loss: {total_loss:.6f}')
        if total_loss < best_train_loss:
            best_train_loss = total_loss
            torch.save(model.state_dict(), f'{model_path}/SDSEF_FM_{file_name}.pth')
            print(f'Saved best model to {model_path}/SDSEF_FM_{file_name}.pth')
            best_model_path = f'{model_path}/SDSEF_FM_{file_name}.pth'
    # 绘制训练损失
    plot_loss(losses, f'{visual_path}/SDSEF_FM_loss_{file_name}.png')
    # 测试
    model.load_state_dict(torch.load(best_model_path))
    scaled_limit = np.array(limit) / scale
    plot_flowfield(model,scaled_limit,positions,f'{visual_path}/SDSEF_FM_flowfield_{file_name}.png')
