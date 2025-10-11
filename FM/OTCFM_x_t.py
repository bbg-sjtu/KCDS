import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import ot as pot
import torch
import torchdyn
from torchdyn.core import NeuralODE
from torchdyn.datasets import generate_moons

from torchcfm.conditional_flow_matching import *
from torchcfm.models.models import *
from Utils import *
from Lasa import ImportLASADataset, filename_list
from EnvelopRegion import TrajectoryRegionClassifier
from torchcfm.optimal_transport import OTPlanSampler
import joblib

def sample_conditional_pt(x0, x1, t, sigma):
    """
    Draw a sample from the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

    Parameters
    ----------
    x0 : Tensor, shape (bs, *dim)
        represents the source minibatch
    x1 : Tensor, shape (bs, *dim)
        represents the target minibatch
    t : FloatTensor, shape (bs)

    Returns
    -------
    xt : Tensor, shape (bs, *dim)

    References
    ----------
    [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
    """
    t = t.reshape(-1, *([1] * (x0.dim() - 1)))
    mu_t = t * x1 + (1 - t) * x0
    epsilon = torch.randn_like(x0)
    return mu_t + sigma * epsilon

def compute_conditional_vector_field(x0, x1):
    """
    Compute the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1].

    Parameters
    ----------
    x0 : Tensor, shape (bs, *dim)
        represents the source minibatch
    x1 : Tensor, shape (bs, *dim)
        represents the target minibatch

    Returns
    -------
    ut : conditional vector field ut(x1|x0) = x1 - x0

    References
    ----------
    [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
    """
    return x1 - x0

def OTCFM_Trainer(file_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ot_sampler = OTPlanSampler(method="exact")
    sigma = 0.01
    batch_size = 256
    # gmm_threshold=12
    model = MLP(dim=2, time_varying=True).to(device)
    # model = BasinLyapunovNet().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    positions, velocities, indices = ImportLASADataset(f'{file_name}.mat')
    # gmm=joblib.load(f'models/GMM/trajectory_gmm_{file_name}.pkl')
    r1 = 2  # 内部区域半径
    r2 = 5   # 边界区域外半径
    regionClassifier = TrajectoryRegionClassifier(positions, r1, r2)

    x_min, x_max = positions[:, 0].min() - 20, positions[:, 0].max() + 20
    y_min, y_max = positions[:, 1].min() - 20, positions[:, 1].max() + 20
    limit=[x_min, x_max, y_min, y_max]

    start = time.time()
    loss_list = []
    for k in range(5000):
        optimizer.zero_grad()

        # 使用GMM拟合轨迹并划分区域，然后采样
        # x0 = sample_source_points(gmm, batch_size, gmm_threshold, limit).to(device)
        # x1 = sample_gmm_points(gmm, batch_size, gmm_threshold, limit).to(device)

        # 使用圆包络直接划分轨迹区域
        x0 = regionClassifier.sample_source_points(batch_size, limit).to(device)
        x1 = regionClassifier.sample_inner_points(batch_size, limit).to(device)

        # Draw samples from OT plan
        x0, x1 = ot_sampler.sample_plan(x0, x1)

        t = torch.rand(x0.shape[0]).type_as(x0)
        xt = sample_conditional_pt(x0, x1, t, sigma=0.01)
        ut = compute_conditional_vector_field(x0, x1)

        # vt = model.compute_velocity(torch.cat([xt, t[:, None]], dim=-1))
        vt = model(torch.cat([xt, t[:, None]], dim=-1))
        loss = torch.mean((vt - ut) ** 2)
        loss_list.append(loss.item())

        loss.backward()
        optimizer.step()

        if (k + 1) % 500 == 0:
            print(f"epoch: {k + 1}, loss: {loss.item()}")

    # 保存模型
    np.savetxt(f"models/OTCFM/loss_{file_name}.txt", loss_list)
    plot_loss(loss_list, f'visualizations/OTCFM/OTCFM_x_t_loss_{file_name}.png')
    torch.save(model.state_dict(), f'models/OTCFM/OTCFM_{file_name}.pth')
    print(f"Model saved to models/OTCFM/OTCFM_{file_name}.pth")

    Nt=100
    Np=1024
    test_points=regionClassifier.sample_source_points(Np, limit).to(device)
    res_xt, res_vt=OTCFM_inference(model,test_points,Nt,Np)
    plot_trajectories(res_xt, f'visualizations/OTCFM/OTCFM_x_t_traj_{file_name}.png')

if __name__ == "__main__":
    for i in range(len(filename_list)):
        print("Trajectory:", filename_list[i])
        OTCFM_Trainer(filename_list[i])