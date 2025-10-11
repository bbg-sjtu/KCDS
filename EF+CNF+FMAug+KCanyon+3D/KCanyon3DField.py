import torch
import torch.nn as nn
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class KCanyon3D(nn.Module):
    def __init__(self, direction: Tuple[float, float], a_init=1.0, b_init=10.0):
        """
        初始化3D单射线K-Canyon势能场
        
        参数:
            direction: 射线方向，为(θ, φ)元组，其中
                       θ是极角（与z轴夹角），φ是方位角（与x轴夹角）
            a_init, b_init: 势能场参数初始值
        """
        super(KCanyon3D, self).__init__()
        
        # 将方向转换为单位向量
        self.direction = self._convert_direction_to_vector(direction)
        
        # 将a和b设置为可训练的参数，并限制在[0, 20]区间内
        self.a_param = nn.Parameter(torch.tensor(a_init, dtype=torch.float32))
        self.b_param = nn.Parameter(torch.tensor(b_init, dtype=torch.float32))
        
        # 注册缓冲区用于存储参数的约束值
        self.register_buffer('a_min', torch.tensor(0.0))
        self.register_buffer('a_max', torch.tensor(20.0))
        self.register_buffer('b_min', torch.tensor(0.0))
        self.register_buffer('b_max', torch.tensor(20.0))
    
    def _convert_direction_to_vector(self, direction):
        """将(θ, φ)角度转换为单位向量"""
        theta, phi = direction
        x = math.sin(theta) * math.cos(phi)
        y = math.sin(theta) * math.sin(phi)
        z = math.cos(theta)
        return torch.tensor([x, y, z], dtype=torch.float32).to(device)
    
    @property
    def a(self):
        # 应用约束，确保a在[0, 20]范围内
        return torch.clamp(self.a_param, self.a_min, self.a_max)
    
    @property
    def b(self):
        # 应用约束，确保b在[0, 20]范围内
        return torch.clamp(self.b_param, self.b_min, self.b_max)

    def smooth_step(self, t, low, high):
        """
        平滑步进函数，在[low, high]之间进行平滑插值。
        当t <= low时返回0，t >= high时返回1，之间返回3x^2 - 2x^3。
        """
        # 确保所有输入都是张量
        t = torch.as_tensor(t, device=device)
        low = torch.as_tensor(low, device=device)
        high = torch.as_tensor(high, device=device)
        
        # 创建掩码
        mask_low = t <= low
        mask_high = t >= high
        mask_mid = ~(mask_low | mask_high)
        
        # 计算结果
        result = torch.zeros_like(t)
        result[mask_low] = 0.0
        result[mask_high] = 1.0
        
        # 中间区域的平滑插值
        if mask_mid.any():
            t_mid = t[mask_mid]
            x = (t_mid - low) / (high - low)
            result[mask_mid] = x * x * (3.0 - 2.0 * x)
        
        return result

    def compute_potential(self, xyz_tensor):
        """
        计算给定点(x,y,z)在单射线流场中的势能值（平滑过渡版本）。
        输入: xyz_tensor - 形状为[batch_size, 3]的张量
        """
        batch_size = xyz_tensor.shape[0]
        
        # 计算点到原点的距离
        r = torch.norm(xyz_tensor, dim=1)
        
        # 处理原点
        origin_mask = r == 0
        non_origin_mask = ~origin_mask
        
        # 初始化结果张量
        potential = torch.zeros(batch_size, dtype=xyz_tensor.dtype, device=xyz_tensor.device)
        
        # 原点处的势能为0
        if origin_mask.any():
            potential[origin_mask] = 0.0
        
        # 非原点处的计算
        if non_origin_mask.any():
            xyz_nonzero = xyz_tensor[non_origin_mask]
            r_nonzero = r[non_origin_mask]
            
            # 计算单位方向向量
            p_hat = xyz_nonzero / r_nonzero.unsqueeze(1)
            
            # 计算点与射线的夹角
            dot_product = torch.sum(p_hat * self.direction, dim=1)
            theta = torch.acos(torch.clamp(dot_product, -1.0, 1.0))
            
            # 定义过渡区域（在π/2附近）
            transition_width = math.pi / 8
            low_angle = math.pi/2 - transition_width
            high_angle = math.pi/2 + transition_width
            
            # 计算权重w
            w = self.smooth_step(theta, low_angle, high_angle)
            
            # 计算两种势能
            direct_potential = 0.5 * self.a * r_nonzero**2
            ray_potential = 0.5 * self.a * r_nonzero**2 + 0.5 * self.b * r_nonzero**2 * theta**2
            
            potential[non_origin_mask] = w * direct_potential + (1 - w) * ray_potential
        
        return potential

    def compute_velocity(self, xyz_tensor):
        """
        计算给定点(x,y,z)在单射线流场中的速度向量。
        使用数值方法计算梯度，避免自动微分的问题。
        输入: xyz_tensor - 形状为[batch_size, 3]的张量
        """
        # 使用数值方法计算梯度
        h = 1e-4  # 微小变化量
        
        # 计算当前点的势能
        U0 = self.compute_potential(xyz_tensor)
        
        # 计算x方向的梯度
        x_plus = xyz_tensor.clone()
        x_plus[:, 0] += h
        U_x_plus = self.compute_potential(x_plus)
        
        x_minus = xyz_tensor.clone()
        x_minus[:, 0] -= h
        U_x_minus = self.compute_potential(x_minus)
        
        dU_dx = (U_x_plus - U_x_minus) / (2 * h)
        
        # 计算y方向的梯度
        y_plus = xyz_tensor.clone()
        y_plus[:, 1] += h
        U_y_plus = self.compute_potential(y_plus)
        
        y_minus = xyz_tensor.clone()
        y_minus[:, 1] -= h
        U_y_minus = self.compute_potential(y_minus)
        
        dU_dy = (U_y_plus - U_y_minus) / (2 * h)
        
        # 计算z方向的梯度
        z_plus = xyz_tensor.clone()
        z_plus[:, 2] += h
        U_z_plus = self.compute_potential(z_plus)
        
        z_minus = xyz_tensor.clone()
        z_minus[:, 2] -= h
        U_z_minus = self.compute_potential(z_minus)
        
        dU_dz = (U_z_plus - U_z_minus) / (2 * h)
        
        # 组合梯度向量
        gradients = torch.stack([dU_dx, dU_dy, dU_dz], dim=1)
        
        # 速度是势能的负梯度
        velocity = -gradients
        
        return velocity


def visualize_3d_field(field, direction, z_slice=None, resolution=30, range_=(-2, 2)):
    """
    可视化3D K-Canyon势能场和速度场
    
    参数:
        field: KCanyon3D_SingleRay实例
        direction: 方向，用于标注
        z_slice: 如果提供，则只绘制该z值的切片；否则绘制3D等势面
        resolution: 网格分辨率
        range_: 坐标范围
    """
    # 创建网格
    x = np.linspace(range_[0], range_[1], resolution)
    y = np.linspace(range_[0], range_[1], resolution)
    
    if z_slice is not None:
        # 2D切片可视化
        X, Y = np.meshgrid(x, y)
        Z = np.full_like(X, z_slice)
        points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        points_tensor = torch.tensor(points, dtype=torch.float32, device=device)
        
        # 计算势能和速度
        with torch.no_grad():
            potential = field.compute_potential(points_tensor).cpu().numpy().reshape(resolution, resolution)
            velocity = field.compute_velocity(points_tensor).cpu().numpy().reshape(resolution, resolution, 3)
        
        # 创建子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 绘制势能场
        im1 = ax1.contourf(X, Y, potential, levels=20, cmap='viridis')
        ax1.set_title(f'Potential Field at z={z_slice}')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        plt.colorbar(im1, ax=ax1)
        
        # 绘制速度场
        speed = np.linalg.norm(velocity, axis=2)
        # 每隔几个点绘制一个箭头，避免过于密集
        step = max(1, resolution // 15)
        im2 = ax2.quiver(
            X[::step, ::step], Y[::step, ::step], 
            velocity[::step, ::step, 0], velocity[::step, ::step, 1], 
            speed[::step, ::step], cmap='cool', scale=20
        )
        ax2.set_title(f'Velocity Field at z={z_slice}')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        plt.colorbar(im2, ax=ax2)
        
        # 标记方向
        theta, phi = direction
        dx = np.sin(theta) * np.cos(phi)
        dy = np.sin(theta) * np.sin(phi)
        dz = np.cos(theta)
        
        # 如果方向接近当前z平面，则标记
        if abs(dz - z_slice) < 0.2:
            ax1.quiver(0, 0, dx, dy, color='red', scale=5)
            ax2.quiver(0, 0, dx, dy, color='red', scale=5)
            ax1.text(dx, dy, 'Ray', color='red')
            ax2.text(dx, dy, 'Ray', color='red')
        
    else:
        # 3D等势面可视化
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 创建3D网格
        x = np.linspace(range_[0], range_[1], resolution)
        y = np.linspace(range_[0], range_[1], resolution)
        z = np.linspace(range_[0], range_[1], resolution)
        X, Y, Z = np.meshgrid(x, y, z)
        
        # 采样点（减少点数以提高性能）
        sample_step = max(1, resolution // 15)
        sample_points = np.stack([
            X[::sample_step, ::sample_step, ::sample_step],
            Y[::sample_step, ::sample_step, ::sample_step],
            Z[::sample_step, ::sample_step, ::sample_step]
        ], axis=-1).reshape(-1, 3)
        
        points_tensor = torch.tensor(sample_points, dtype=torch.float32, device=device)
        
        # 计算势能
        with torch.no_grad():
            potential = field.compute_potential(points_tensor).cpu().numpy()
        
        # 绘制等势面
        # 选择几个等势面值
        isovalues = np.linspace(potential.min(), potential.max(), 5)[1:-1]
        
        for i, isovalue in enumerate(isovalues):
            # 找到接近等势面的点
            mask = np.abs(potential - isovalue) < 0.1
            if np.any(mask):
                points_near_isosurface = sample_points[mask]
                ax.scatter(
                    points_near_isosurface[:, 0],
                    points_near_isosurface[:, 1],
                    points_near_isosurface[:, 2],
                    c=potential[mask],
                    cmap='viridis',
                    alpha=0.3,
                    s=10
                )
        
        # 标记方向
        theta, phi = direction
        dx = np.sin(theta) * np.cos(phi)
        dy = np.sin(theta) * np.sin(phi)
        dz = np.cos(theta)
        
        ax.quiver(0, 0, 0, dx, dy, dz, color='red', length=1.5, arrow_length_ratio=0.1)
        ax.text(dx, dy, dz, 'Ray', color='red')
        
        ax.set_title('3D Potential Isosurfaces')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
    plt.tight_layout()
    plt.show()

def visualize_trajectories(field, start_points, num_steps=300, dt=0.01):
    """
    可视化从不同起点开始的轨迹
    
    参数:
        field: KCanyon3D_SingleRay实例
        start_points: 起点列表
        num_steps: 模拟步数
        dt: 时间步长
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(start_points)))
    
    for i, start_point in enumerate(start_points):
        # 初始化轨迹
        trajectory = [start_point]
        current_point = torch.tensor(start_point, dtype=torch.float32, device=device).unsqueeze(0)
        
        # 模拟轨迹
        for step in range(num_steps):
            # 计算速度
            with torch.no_grad():
                velocity = field.compute_velocity(current_point)
            
            # 更新位置（欧拉积分）
            current_point = current_point + velocity * dt
            
            # 如果接近原点，停止模拟
            if torch.norm(current_point) < 0.1:
                break
            
            trajectory.append(current_point.squeeze().cpu().numpy())
        
        trajectory = np.array(trajectory)
        
        # 绘制轨迹
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                color=colors[i], linewidth=2, label=f'Start {i+1}')
        ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
                  color=colors[i], s=50, marker='o')
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], 
                  color=colors[i], s=50, marker='x')
    
    ax.set_title('Trajectories in the 3D K-Canyon Field')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

# 使用示例
if __name__ == "__main__":
    # 定义6个方向（正立方体的6个面方向）
    directions = (np.pi/2, 0)               # +Z方向
    
    # 创建势能场
    field = KCanyon3D(directions)
    
    # 可视化轨迹
    start_points = [
        [0.0, 0.0, 2.0],
        [0.0, 1.0, 2.0],
        [0.0, -1.0, 2.0],
        [1.0, 1.0, 2.0],
        [1.0, -1.0, 2.0],
        [1.0, 0.0, 2.0],
        [-1.0, 0.0, 2.0],
        [0.0, 1.0, 2.0],
        [0.0, -1.0, 2.0],
        [2.0, 0.0, -2],
        [0.0, 2.0, -2],
        [0.0, 0.0, -2],
    ]
    
    visualize_trajectories(field, start_points)