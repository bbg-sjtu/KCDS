# 使用FM数据增强,训练CanyonVF其中两个参数,训练出稳定收缩行为
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.preprocessing import MinMaxScaler
import warnings
from Lasa import ImportLASADataset, file_list1_1
from torch.utils.data import Dataset, DataLoader
from utils import *
from Dataset import *
from Canyontrainable_models_3d import *
import os
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(42)

dim=2
h=64
l=5
K=2
lr=0.01
weight_decay=1e-8
num_epochs=31

model_save_path = f"models/SDSEF_FM_KCanyon_h{h}_l{l}_k{K}"
visual_path = f"visualizations/SDSEF_FM_KCanyon_h{h}_l{l}_k{K}"
os.makedirs(model_save_path, exist_ok=True)
os.makedirs(visual_path, exist_ok=True)

# for file_name in file_list1_1:

file_name='Multi_Models_3Py'
print("Trajectory:", file_name)

dataset, loader, positions, limit = generate_global_augment_dataset(file_name)
x0s=torch.from_numpy(positions[[0, 1000, 2000, 3000, 4000, 5000, 6000],:]).to(device, dtype=torch.float32)
x_star=torch.tensor([[0,0]]).to(device, dtype=torch.float32)

# 初始化模型
# MLP
SDSEF_FM_velocity_modelpath=f"models/SDSEF_FM_h{h}_l{l}_Velocity_Models/velocity_model_h{h}_l{l}_{file_name}.pth"
model = SDSEF_FM_KCanyonTrainable_3D(dim=dim, hidden_dim=h, num_layers=l, K=K, SDSEF_FM_modelpath=SDSEF_FM_velocity_modelpath, x0s=x0s, x_star=x_star).to(device)

optimizer = optim.Adam(model.CanyonVF.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
criterion = nn.MSELoss()

plot_potential_velocity_field(model, limit, positions, f'{visual_path}/SDSEF_FM_KCanyon_p_v_{file_name}_{0}.png')
print(f'a: {model.CanyonVF.a_param}. b: {model.CanyonVF.b_param}.')

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

        # # 计算损失
        # loss = criterion(pred_velocities, v)
        # 只计算方向上的损失,而不考虑速度大小
        # 归一化真实速度和预测速度（只保留方向信息）
        v_normalized = F.normalize(v, p=2, dim=1)
        pred_normalized = F.normalize(pred_velocities, p=2, dim=1)
        
        # 计算余弦相似度损失（1 - 余弦相似度）
        cosine_sim = F.cosine_similarity(pred_normalized, v_normalized, dim=1)
        loss = 1 - cosine_sim.mean()
        
        # 反向传播
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    losses.append(total_loss)
    print(f'Epoch {epoch}, Loss: {total_loss:.6f}')
    scheduler.step()
    if total_loss < best_train_loss:
        best_train_loss = total_loss
        torch.save(model.state_dict(), f'{model_save_path}/SDSEF_FM_KCanyon_{file_name}.pth')
        print(f'Saved best model to {model_save_path}/SDSEF_FM_KCanyon_{file_name}.pth')
        best_model_path = f'{model_save_path}/SDSEF_FM_KCanyon_{file_name}.pth'
    
    if (epoch+1) % 10 == 0:
        torch.save(model.state_dict(), f'{model_save_path}/SDSEF_FM_KCanyon_{file_name}_{epoch+1}.pth')
        plot_potential_velocity_field(model, limit, positions, f'{visual_path}/SDSEF_FM_KCanyon_p_v_{file_name}_{epoch+1}.png')
        print(f'a: {model.CanyonVF.a_param}. b: {model.CanyonVF.b_param}.')
# 绘制训练损失
plot_loss(losses, f'{visual_path}/SDSEF_FM_KCanyon_loss_{file_name}.png')
# # 测试
# model.load_state_dict(torch.load(best_model_path))
# print(f'a: {model.CanyonVF.a_param}. b: {model.CanyonVF.b_param}.')

