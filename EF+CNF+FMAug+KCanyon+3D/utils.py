from turtle import speed
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.preprocessing import MinMaxScaler
from Lasa import ImportLASADataset, filename_list
from matplotlib import cm

from models import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)


def prepare_demonstrations(demo):
    """
    准备演示数据，将其归一化到 [-0.5, 0.5] 范围
    保持原始零点位置不变，并相应调整速度数据
    """
    # 获取位置数据
    positions = demo['positions']
    
    # 计算最大绝对值范围，用于对称缩放
    max_abs = np.max(np.abs(positions), axis=0)
    # 避免除零错误，确保至少有一个非零范围
    max_abs = np.maximum(max_abs, 1e-8)
    # 计算缩放因子，使数据范围在[-0.5, 0.5]之间
    scale = 2 * max_abs
    
    # 归一化位置数据：先除以缩放因子，然后偏移到[-0.5, 0.5]范围
    normalized_positions = positions / scale
    
    # 调整速度数据：速度需要同样的缩放因子
    normalized_velocities = demo['velocities'] / scale
    
    normalized_demos={
        'positions': normalized_positions,
        'velocities': normalized_velocities,
        'target': demo['target']
    }
    
    return normalized_demos, scale

def create_training_data(demo):
    """
    从归一化的演示创建训练数据
    """
    
    positions = demo['positions']
    velocities = demo['velocities']
    targets = np.repeat([demo['target']], len(demo['positions']), axis=0)

    return torch.FloatTensor(positions), torch.FloatTensor(velocities), torch.FloatTensor(targets)


def generate_trajectory(model, start_point, target_point, num_steps=100, dt=0.01):
    """
    使用学习到的动力学从起点到目标生成轨迹
    """
    trajectory = [start_point]
    current_point = start_point.clone()
    
    for _ in range(num_steps):
        # 计算当前点的速度
        with torch.no_grad():
            velocity = model.direct_velocity_prediction(
                current_point.unsqueeze(0), 
                target_point.unsqueeze(0)
            )
        
        # 更新位置
        current_point = current_point + velocity.squeeze(0) * dt
        
        # 检查是否接近目标
        if torch.norm(current_point - target_point) < 1e-3:
            break
            
        trajectory.append(current_point.clone())
    
    return torch.stack(trajectory)

def test_diffeo(model):
    # 测试微分同胚映射
    test_points = torch.FloatTensor([
        [-0.4, 0.4],
        [-0.4, -0.4],
        [0.4, 0.4],
        [0.4, -0.4]
    ]).to(device)

    # 应用前向变换
    transformed_points = model.diffeo.differentiable_forward(test_points)
    # 应用逆变换
    reconstructed_points = model.diffeo.differentiable_inverse(transformed_points)
    print("Original points:", test_points)
    print("Transformed points:", transformed_points)
    print("Reconstructed points:", reconstructed_points)
    print("Reconstruction error:", torch.norm(test_points - reconstructed_points).item())

    # 应用前向变换
    transformed_points = model.diffeo.forward(test_points)
    # 应用逆变换
    reconstructed_points = model.diffeo.inverse(transformed_points)
    print("Original points:", test_points)
    print("Transformed points:", transformed_points)
    print("Reconstructed points:", reconstructed_points)
    print("Reconstruction error:", torch.norm(test_points - reconstructed_points).item())

def plot_flowfield(model,limit,positions,savePath):
    # 创建网格 - 确保X和Y形状相同
    x_grid = np.linspace(limit[0,0], limit[1,0], 100)
    y_grid = np.linspace(limit[0,1], limit[1,1], 100)
    X_np, Y_np = np.meshgrid(x_grid, y_grid)  # 这会创建相同形状的网格
    
    # 将网格点展平用于计算
    grid = torch.tensor(np.column_stack([X_np.flatten(), Y_np.flatten()]), dtype=torch.float32).to(device)
    
    # 计算空间的速度场
    target=torch.zeros_like(grid).to(device)
    # with torch.no_grad():
    f_orig = model.direct_velocity_prediction(grid, target)

    # 重塑速度场为网格形状
    U_orig = f_orig[:, 0].reshape(X_np.shape).detach().cpu().numpy()
    V_orig = f_orig[:, 1].reshape(Y_np.shape).detach().cpu().numpy()

    # 绘制原始空间流线图
    plt.figure(figsize=(10, 8))
    plt.streamplot(X_np, Y_np, U_orig, V_orig, density=1.5, color='blue', linewidth=1)
    
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

    plt.title("Velocity Field")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(limit[0,0], limit[1,0])
    plt.ylim(limit[0,1], limit[1,1])
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


def plot_potential_velocity_field(model, scaled_limit, positions, savePath=None):
    # 创建网格点
    x = np.linspace(scaled_limit[0,0], scaled_limit[1,0], 50)
    y = np.linspace(scaled_limit[0,1], scaled_limit[1,1], 50)

    X, Y = np.meshgrid(x, y)
    grid_points = np.column_stack([X.flatten(), Y.flatten()])
    grid_tensor = torch.FloatTensor(grid_points).to(device)

    target_point=np.array([0, 0])
    target_tensor = torch.FloatTensor(target_point).unsqueeze(0).to(device)

    y_points = model.diffeo.differentiable_forward(grid_tensor)
    y_target = model.diffeo.differentiable_forward(target_tensor)
    diff = y_points - y_target
    potential=model.CanyonVF.compute_potential(diff).detach().cpu().numpy()
    velocities = model.direct_velocity_prediction(grid_tensor, target_tensor).detach().cpu().numpy()


    # 重塑为网格形状
    Z = potential.reshape(X.shape)

    # 创建图形
    fig = plt.figure(figsize=(15, 10))

    # 1. 三维表面图
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=True, alpha=0.8)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Potential')
    ax1.set_title('3D Potential Field')

    # 为3D图添加colorbar，调整尺寸
    cbar1 = fig.colorbar(surf, ax=ax1, shrink=0.6, aspect=20, pad=0.1)
    cbar1.ax.set_ylabel('Potential', rotation=270, labelpad=15)


    U = velocities[:, 0].reshape(X.shape)
    V = velocities[:, 1].reshape(Y.shape)
    speed = np.sqrt(U**2 + V**2)

    # 创建图形
    ax2 = fig.add_subplot(1, 2, 2)

    contour = ax2.contourf(X, Y, Z, 20, cmap=cm.viridis, alpha=0.7)
    stream = ax2.streamplot(X, Y, U, V, density=1.5, color=speed, cmap=plt.cm.jet, linewidth=1, arrowsize=1)

    tmpx = positions
    for i in range(7):
        plt.plot(tmpx[1000*i:1000*(i+1), 0], tmpx[1000*i:1000*(i+1), 1], color='red', linewidth=2, alpha=0.7)
        # plt.plot(reproduction[:, i, 0], reproduction[:, i, 1], color='white', linestyle='--', linewidth=1, alpha=0.7)
    # 标记起点
    for i in range(7):
        start_point = tmpx[1000*i]
        plt.scatter(start_point[0], start_point[1], s=100, c='red', marker='o', edgecolors='black', zorder=5)
    # 标记终点
    for i in range(7):
        end_point = tmpx[1000*(i+1)-1]
        plt.scatter(end_point[0], end_point[1], s=100, c='red', marker='s', edgecolors='black', zorder=5)

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Potential Field with Flow Streamlines')
    ax2.set_aspect('equal')
    fig.colorbar(stream.lines, ax=ax2, label='Velocity Magnitude')


    # 标记目标点
    ax2.plot(target_point[0], target_point[1], 'ro', markersize=10, label='Target')
    plt.xlim(scaled_limit[0,0], scaled_limit[1,0])
    plt.ylim(scaled_limit[0,1], scaled_limit[1,1])

    # 调整布局，确保两个子图和colorbar有足够的空间
    plt.tight_layout()

    # plt.show()
    plt.savefig(savePath, dpi=300)
    print(f'Saved visualization to {savePath}')
    plt.close()

def plot_inference(model, positions, limit):

    target_point=np.array([0, 0])
    target_tensor = torch.FloatTensor(target_point).unsqueeze(0).to(device)

    startpoints=positions[[0, 1000, 2000, 3000, 4000, 5000, 6000],:].to(device)
    reproduction=model.inference(startpoints, target_tensor, dt=0.01, steps=500)
    plt.figure(figsize=(10, 8))
    for i in range(7):
        plt.plot(positions[1000*i:1000*(i+1), 0], positions[1000*i:1000*(i+1), 1], color='red', linewidth=2, alpha=0.7)
        plt.plot(reproduction[:, i, 0], reproduction[:, i, 1], color='black', linestyle='--', linewidth=2, alpha=0.7)

    plt.show()