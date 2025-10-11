from fileinput import filename
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchdyn
from torchdyn.datasets import generate_moons
from scipy.spatial import cKDTree

# Implement some helper functions
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):
        return self.model(torch.cat([x, t.repeat(x.shape[0])[:, None]], 1))

def OTCFM_inference(model,test_points,Nt,Np):
    dt=1/Nt
    xt=test_points
    res_xt = np.zeros((Np, Nt, 2))
    res_vt= np.zeros((Np, Nt, 2))
    for j in range(Nt):
        t = torch.full((Np,), j * dt).to(device)
        vt = model(torch.cat([xt, t[:, None]], dim=-1))
        xt=xt+vt*dt
        res_xt[:, j, :] = xt.detach().cpu().numpy()
        res_vt[:, j, :] = vt.detach().cpu().numpy()
    return res_xt, res_vt

def OTCFM_x_inference(model,test_points,Nt,Np):
    dt=0.01
    xt=test_points
    res_xt = np.zeros((Np, Nt, 2))
    res_vt= np.zeros((Np, Nt, 2))
    for j in range(Nt):
        # t = torch.full((Np,), j * dt).to(device)
        vt = model(xt)
        xt=xt+vt*dt
        res_xt[:, j, :] = xt.detach().cpu().numpy()
        res_vt[:, j, :] = vt.detach().cpu().numpy()
    return res_xt, res_vt

def plot_trajectories(traj, limit, positions, savePath, plotboundary=False):
    """Plot trajectories of some selected samples."""
    plt.figure(figsize=(10, 8))
    plt.scatter(traj[:, 0, 0], traj[:, 0, 1], s=10, alpha=0.8, c="black")
    plt.scatter(traj[:, :, 0], traj[:, :, 1], s=0.2, alpha=0.2, c="olive")
    plt.scatter(traj[:, -1, 0], traj[:, -1, 1], s=4, alpha=1, c="blue")

    if plotboundary:

        # 创建网格 - 确保X和Y形状相同
        x_grid = np.linspace(limit[0], limit[1], 100)
        y_grid = np.linspace(limit[2], limit[3], 100)
        X_np, Y_np = np.meshgrid(x_grid, y_grid)  # 这会创建相同形状的网格
        # 创建轨迹的KDTree用于快速距离计算
        tree = cKDTree(positions)
        grid_points = np.column_stack([X_np.ravel(), Y_np.ravel()])
        
        # 计算每个网格点到最近轨迹点的距离
        dists, _ = tree.query(grid_points, k=1)
        dists = dists.reshape(X_np.shape)
        
        # 绘制r=r1的边界线 (使用contour绘制等值线)
        r1=0.05
        contour = plt.contour(X_np, Y_np, dists, levels=[r1], colors='purple', linestyles='dashed', linewidths=2)
        plt.clabel(contour, inline=True, fontsize=10, fmt=f'r = {r1:.2f}')

    plt.legend(["x0", "Flow", "x1"])
    plt.title("Inference Trajectories")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(limit[0], limit[1])
    plt.ylim(limit[2], limit[3])
    plt.savefig(savePath, dpi=300)
    print(f'Saved visualization to {savePath}')
    plt.close()

def plot_loss(loss_list, savePath):
    """Plot loss."""
    plt.figure(figsize=(10,8))
    plt.plot(loss_list, color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    # plt.show()
    plt.savefig(savePath, dpi=300)
    print(f'Saved visualization to {savePath}')
    plt.close()

def plot_flowfield(model,limit,positions,savePath, plotboundary=False, plotmin=False, plottraj=True, plotcircle=False):
    # 创建网格 - 确保X和Y形状相同
    x_grid = np.linspace(limit[0], limit[1], 200)
    y_grid = np.linspace(limit[2], limit[3], 200)
    X_np, Y_np = np.meshgrid(x_grid, y_grid)  # 这会创建相同形状的网格
    
    # 将网格点展平用于计算
    grid = torch.tensor(np.column_stack([X_np.flatten(), Y_np.flatten()]), dtype=torch.float32).to(device)
    
    # 计算空间的速度场
    f_orig = model(grid)
    
    # 重塑速度场为网格形状
    U_orig = f_orig[:, 0].reshape(X_np.shape).detach().cpu().numpy()
    V_orig = f_orig[:, 1].reshape(Y_np.shape).detach().cpu().numpy()

    # 绘制原始空间流线图
    plt.figure(figsize=(10, 8))
    plt.streamplot(X_np, Y_np, U_orig, V_orig, density=1.5, color='blue', linewidth=1)
    
    if plottraj:
        tmpx = positions
        for i in range(7):
            plt.plot(tmpx[1000*i:1000*(i+1), 0], tmpx[1000*i:1000*(i+1), 1], color='red', linewidth=2, alpha=0.7, label=f'Trajectory {i+1}' if i == 0 else "")
        # 标记起点
        for i in range(7):
            start_point = tmpx[1000*i]
            plt.scatter(start_point[0], start_point[1], s=100, c='red', marker='o', edgecolors='black', zorder=5)
        # 标记终点
        for i in range(7):
            end_point = tmpx[1000*(i+1)-1]
            plt.scatter(end_point[0], end_point[1], s=100, c='red', marker='s', edgecolors='black', zorder=5)
    
    # 计算速度大小并找到零速度点
    speed = np.sqrt(U_orig**2 + V_orig**2)
    min_speed_idx = np.unravel_index(np.argmin(speed), speed.shape)  # 找到最小值的二维索引

    if plotmin:
        # 标记零速度点为绿色五角星 ★
        plt.scatter(X_np[min_speed_idx], Y_np[min_speed_idx], 
                s=200,  # 适当调大尺寸
                c='limegreen',  # 明亮的绿色
                marker='*',  # 五角星标记
                edgecolors='darkgreen',  # 深绿色边缘
                linewidths=0.5,  # 边缘线宽
                zorder=5)  # 确保在流线之上
    
    if plotboundary:
        # 创建轨迹的KDTree用于快速距离计算
        tree = cKDTree(positions)
        grid_points = np.column_stack([X_np.ravel(), Y_np.ravel()])
        
        # 计算每个网格点到最近轨迹点的距离
        dists, _ = tree.query(grid_points, k=1)
        dists = dists.reshape(X_np.shape)
        
        # 绘制r=r1的边界线 (使用contour绘制等值线)
        r1=2
        contour = plt.contour(X_np, Y_np, dists, levels=[r1], colors='purple', linestyles='dashed', linewidths=2)
        plt.clabel(contour, inline=True, fontsize=10, fmt=f'r = {r1:.2f}')
    
    if plotcircle:
        # 补充绘制(0,0)附近r1的圆边界
        circle = plt.Circle((0, 0), 5, color='green', fill=False, linestyle='dashed', linewidth=2)
        plt.gca().add_patch(circle)


    plt.title("Velocity Field")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(limit[0], limit[1])
    plt.ylim(limit[2], limit[3])
    plt.savefig(savePath, dpi=300)
    print(f'Saved visualization to {savePath}')
    plt.close()

def plot_boundary(trajectory_points, limit, save_path):
    """
    绘制示教轨迹及其r=r1的边界线
    
    参数:
        trajectory_points: 轨迹点数组, 形状为 (N, 2)
        r1: 边界半径
        limit: 绘图范围 [xmin, xmax, ymin, ymax]
        save_path: 图片保存路径 (可选)
        resolution: 网格分辨率 (点数)
    """
    # 创建轨迹的KDTree用于快速距离计算
    tree = cKDTree(trajectory_points)
    
    # 创建网格
    x_grid = np.linspace(limit[0], limit[1], 100)
    y_grid = np.linspace(limit[2], limit[3], 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    
    # 计算每个网格点到最近轨迹点的距离
    dists, _ = tree.query(grid_points, k=1)
    dists = dists.reshape(X.shape)
    
    # 创建图形
    plt.figure(figsize=(10, 8))
    
    # 绘制轨迹
    plt.plot(trajectory_points[:, 0], trajectory_points[:, 1], 
             'b-', linewidth=2, alpha=0.7, label='Trajectory')
    
    # 绘制起点和终点
    plt.scatter(trajectory_points[0, 0], trajectory_points[0, 1], 
                s=100, c='green', marker='o', edgecolors='black', zorder=5, label='Start')
    plt.scatter(trajectory_points[-1, 0], trajectory_points[-1, 1], 
                s=100, c='red', marker='s', edgecolors='black', zorder=5, label='End')
    
    # 绘制r=r1的边界线 (使用contour绘制等值线)
    r1=2
    contour = plt.contour(X, Y, dists, levels=[r1], colors='purple', linestyles='dashed', linewidths=2)
    plt.clabel(contour, inline=True, fontsize=10, fmt=f'r = {r1:.2f}')
    
    # 添加标题和图例
    plt.title(f'Trajectory with Boundary (r1={r1})')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    
    # 设置坐标轴范围
    plt.xlim(limit[0], limit[1])
    plt.ylim(limit[2], limit[3])
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_combined_flowfield(innerModel, outerModel, classifier, positions, r1, r2, limit, savePath, plotboundary=True, plotmin=True, plottraj=True):

    # 创建网格
    x_grid = np.linspace(limit[0], limit[1], 200)
    y_grid = np.linspace(limit[2], limit[3], 200)
    X_np, Y_np = np.meshgrid(x_grid, y_grid)
    grid_points = np.column_stack([X_np.flatten(), Y_np.flatten()])
    
    # 转换为PyTorch张量
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32).to(device)
    
    # 计算内外部模型的速度预测
    f_inner = innerModel(grid_tensor).detach().cpu().numpy()
    f_outer = outerModel(grid_tensor).detach().cpu().numpy()
    
    inner_mask, boundary_mask, outer_mask, dists = classifier.classify_points(grid_points)
    
    # 初始化组合速度场
    combined_velocity = np.zeros_like(f_inner)
    
    # 内部区域使用innerModel
    combined_velocity[inner_mask] = f_inner[inner_mask]
    
    # 外部区域使用outerModel
    combined_velocity[outer_mask] = f_outer[outer_mask]
    
    # 边界区域使用加权混合
    for i in np.where(boundary_mask)[0]:
        r = dists[i]
        # 计算权重 (r1到r2之间从0到1线性变化)
        alpha = (r - r1) / (r2 - r1)
        # 加权组合
        combined_velocity[i] = alpha * f_outer[i] + (1 - alpha) * f_inner[i]
    
    # 重塑为网格形状
    U_combined = combined_velocity[:, 0].reshape(X_np.shape)
    V_combined = combined_velocity[:, 1].reshape(Y_np.shape)
    
    # 创建图形
    plt.figure(figsize=(10, 8))
    
    # 绘制组合速度流线图
    plt.streamplot(X_np, Y_np, U_combined, V_combined, density=1.5, 
                  color='blue', linewidth=1, arrowsize=1)
    if plottraj:
        tmpx = positions
        for i in range(7):
            plt.plot(tmpx[1000*i:1000*(i+1), 0], tmpx[1000*i:1000*(i+1), 1], color='red', linewidth=2, alpha=0.7, label=f'Trajectory {i+1}' if i == 0 else "")
        # 标记起点
        for i in range(7):
            start_point = tmpx[1000*i]
            plt.scatter(start_point[0], start_point[1], s=100, c='red', marker='o', edgecolors='black', zorder=5)
        # 标记终点
        for i in range(7):
            end_point = tmpx[1000*(i+1)-1]
            plt.scatter(end_point[0], end_point[1], s=100, c='red', marker='s', edgecolors='black', zorder=5)
    
    # 计算速度大小并找到零速度点
    speed = np.sqrt(U_combined**2 + V_combined**2)
    min_speed_idx = np.unravel_index(np.argmin(speed), speed.shape)  # 找到最小值的二维索引

    if plotmin:
        # 标记零速度点为绿色五角星 ★
        plt.scatter(X_np[min_speed_idx], Y_np[min_speed_idx], 
                s=200,  # 适当调大尺寸
                c='limegreen',  # 明亮的绿色
                marker='*',  # 五角星标记
                edgecolors='darkgreen',  # 深绿色边缘
                linewidths=0.5,  # 边缘线宽
                zorder=5)  # 确保在流线之上
    
    if plotboundary:
        # 创建轨迹的KDTree用于快速距离计算
        tree = cKDTree(positions)
        grid_points = np.column_stack([X_np.ravel(), Y_np.ravel()])
        
        # 计算每个网格点到最近轨迹点的距离
        dists, _ = tree.query(grid_points, k=1)
        dists = dists.reshape(X_np.shape)
        
        # 绘制r=r1的边界线 (使用contour绘制等值线)
        # r1=2
        contour = plt.contour(X_np, Y_np, dists, levels=[r1], colors='purple', linestyles='dashed', linewidths=2)
        plt.clabel(contour, inline=True, fontsize=10, fmt=f'r = {r1:.2f}')


    plt.title("Combined Velocity Field")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(limit[0], limit[1])
    plt.ylim(limit[2], limit[3])
    plt.savefig(savePath, dpi=300)
    print(f'Saved visualization to {savePath}')
    plt.close()