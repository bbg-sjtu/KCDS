# 设计了具有K个山谷的势能场，具备唯一稳定点，各个山脊被削为简单直线收敛，控制全局快速收敛行为
# 山谷数量和位置由Euclideanizing Flows中标准场y空间中的直线分布决定

import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class KCanyon_Combined_Field(nn.Module):
    def __init__(self, thetas, a_init=1.0, b_init=10.0):
        super(KCanyon_Combined_Field, self).__init__()
        self.thetas = torch.from_numpy(thetas).to(device)
        self.K = len(thetas)
        
        # 将 a 和 b 设置为可训练的参数，并限制在 [0, 20] 区间内
        self.a_param = nn.Parameter(torch.tensor(a_init, dtype=torch.float32))
        self.b_param = nn.Parameter(torch.tensor(b_init, dtype=torch.float32))
        
        # 注册缓冲区用于存储参数的约束值
        self.register_buffer('a_min', torch.tensor(0.0))
        self.register_buffer('a_max', torch.tensor(20.0))
        self.register_buffer('b_min', torch.tensor(0.0))
        self.register_buffer('b_max', torch.tensor(20.0))
    
    @property
    def a(self):
        # 应用约束，确保 a 在 [0, 20] 范围内
        return torch.clamp(self.a_param, self.a_min, self.a_max)
    
    @property
    def b(self):
        # 应用约束，确保 b 在 [0, 20] 范围内
        return torch.clamp(self.b_param, self.b_min, self.b_max)

    def smooth_step(self, t, low, high):
        """
        平滑步进函数，在[low, high]之间进行平滑插值。
        当t <= low时返回0，t >= high时返回1，之间返回3x^2 - 2x^3。
        """
        # 确保所有输入都是张量
        t = torch.as_tensor(t)
        low = torch.as_tensor(low)
        high = torch.as_tensor(high)
        
        # 创建掩码
        mask_low = t <= low
        mask_high = t >= high
        mask_mid = ~(mask_low | mask_high)
        
        # 计算结果
        result = torch.zeros_like(t)
        result[mask_low] = 0.0
        result[mask_high] = 1.0
        
        # 中间区域的平滑插值
        # x = (t[mask_mid] - low) / (high - low)
        x = (t - low) / (high - low)
        result[mask_mid] = x[mask_mid] * x[mask_mid] * (3.0 - 2.0 * x[mask_mid])
        
        return result

    def compute_potential(self, xy_tensor):
        """
        计算给定点(x,y)在K条射线流场中的势能值（平滑过渡版本）。
        输入: xy_tensor - 形状为[batch_size, 2]的张量
        """
        batch_size = xy_tensor.shape[0]
        x = xy_tensor[:, 0]
        y = xy_tensor[:, 1]
        
        # 处理原点
        origin_mask = (x == 0) & (y == 0)
        non_origin_mask = ~origin_mask
        
        # 初始化结果张量
        potential = torch.zeros(batch_size, dtype=xy_tensor.dtype, device=xy_tensor.device)
        
        # 原点处的势能为0
        if origin_mask.any():
            potential[origin_mask] = 0.0
        
        # 非原点处的计算
        if non_origin_mask.any():
            x_nonzero = x[non_origin_mask]
            y_nonzero = y[non_origin_mask]
            
            r = torch.sqrt(x_nonzero**2 + y_nonzero**2)
            phi_ = torch.atan2(y_nonzero, x_nonzero)
            
            # 将角度归一化到[0, 2π)
            phi = torch.where(phi_ < 0, phi_ + 2 * math.pi, phi_)
            
            if self.K == 1:
                
                # 计算反向角度（直接收敛区域）
                theta_ray = self.thetas
                reverse_theta = (theta_ray + math.pi) % (2 * math.pi)
                
                transition_width = math.pi / 8
                low_angle = reverse_theta - transition_width
                high_angle = reverse_theta + transition_width
                
                # 计算角度差（考虑周期性）
                diff_to_reverse = torch.abs(phi - reverse_theta)
                diff_to_reverse = torch.min(diff_to_reverse, 2 * math.pi - diff_to_reverse)
                
                w = torch.where(
                    diff_to_reverse <= transition_width,
                    torch.tensor(1.0, dtype=xy_tensor.dtype, device=xy_tensor.device),
                    torch.tensor(0.0, dtype=xy_tensor.dtype, device=xy_tensor.device)
                )
                
                # 对于过渡区域，使用平滑过渡
                transition_low = transition_width
                transition_high = 2 * transition_width
                
                transition_mask = (diff_to_reverse > transition_width) & (diff_to_reverse < 2 * transition_width)
                if transition_mask.any():
                    w_transition = 1 - self.smooth_step(
                        diff_to_reverse[transition_mask], 
                        transition_low, 
                        transition_high
                    ).to(torch.float32)
                    w[transition_mask] = w_transition
                
                direct_potential = 0.5 * self.a * r**2
                delta_phi = phi_ - theta_ray
                delta_phi = (delta_phi + math.pi) % (2 * math.pi) - math.pi
                ray_potential = (0.5 * self.a * r**2 + 0.5 * self.b * r**2 * delta_phi**2).to(torch.float32)

                potential[non_origin_mask] = w * direct_potential + (1 - w) * ray_potential

            else:
                # 计算所有射线角度
                # thetas_rays = torch.tensor([i * 2 * math.pi / K for i in range(K)], 
                #                         dtype=xy_tensor.dtype, device=xy_tensor.device)
                thetas_rays=self.thetas
                # 计算所有中间角度
                # thetas_middle = torch.tensor([(2 * i + 1) * math.pi / self.K for i in range(self.K)], 
                #                             dtype=xy_tensor.dtype, device=xy_tensor.device)
                
                # 对射线角度排序并考虑周期性
                sorted_thetas = torch.sort(self.thetas).values
                # 计算相邻射线之间的中间角度（包括首尾之间的）
                thetas_middle = (sorted_thetas + torch.roll(sorted_thetas, -1)) / 2
                # 处理首尾之间的中间角度
                thetas_middle[-1] = (sorted_thetas[-1] + sorted_thetas[0] + 2 * math.pi) / 2
                thetas_middle = thetas_middle % (2 * math.pi)

                # 计算与所有中间角度的最小差值
                phi_expanded = phi.unsqueeze(1)  # [batch_nonzero, 1]
                diffs = torch.abs(phi_expanded - thetas_middle)
                diffs = torch.min(diffs, 2 * math.pi - diffs)
                min_diff_m, min_middle_indices = torch.min(diffs, dim=1)
                
                # 计算与所有射线角度的最小差值并找到最近的射线
                diffs = torch.abs(phi_expanded - thetas_rays)
                diffs = torch.min(diffs, 2 * math.pi - diffs)
                min_diff_r, min_indices = torch.min(diffs, dim=1)
                theta_r_star = thetas_rays[min_indices]
                
                # 计算权重w
                # 计算相邻射线之间的角度间隔
                angles_diff = torch.diff(sorted_thetas, append=sorted_thetas[0].unsqueeze(0) + 2 * math.pi)
                angles_diff = angles_diff % (2 * math.pi)

                point_intervals = angles_diff[min_middle_indices]

                # 基于实际间隔计算阈值
                inner_angle = point_intervals / 8
                outer_angle = point_intervals / 4
                
                w = torch.where(
                    min_diff_m <= inner_angle,
                    torch.tensor(1.0, dtype=xy_tensor.dtype, device=xy_tensor.device),
                    torch.where(
                        min_diff_m >= outer_angle,
                        torch.tensor(0.0, dtype=xy_tensor.dtype, device=xy_tensor.device),
                        1 - self.smooth_step(min_diff_m, inner_angle, outer_angle)
                    )
                )
                
                direct_potential = 0.5 * self.a * r**2
                delta_phi = phi - theta_r_star
                delta_phi = (delta_phi + math.pi) % (2 * math.pi) - math.pi
                ray_potential = (0.5 * self.a * r**2 + 0.5 * self.b * r**2 * delta_phi**2).to(torch.float32)

                potential[non_origin_mask] = w * direct_potential + (1 - w) * ray_potential
        
        return potential

    def compute_velocity(self, xy_tensor):
        """
        计算给定点(x,y)在K条射线流场中的速度向量。
        优化版本：在射线收敛区和直线收敛区使用解析解，过渡区使用自动求导。
        输入: xy_tensor - 形状为[batch_size, 2]的张量
        """
        batch_size = xy_tensor.shape[0]
        x = xy_tensor[:, 0]
        y = xy_tensor[:, 1]
        
        # 处理原点
        origin_mask = (x == 0) & (y == 0)
        non_origin_mask = ~origin_mask
        
        # 初始化结果张量
        velocity = torch.zeros_like(xy_tensor)
        region_type = torch.zeros_like(x)

        # 原点处的速度为0
        if origin_mask.any():
            velocity[origin_mask] = 0.0
        
        # 非原点处的计算
        if non_origin_mask.any():
            # 获取非原点点的索引
            non_origin_indices = torch.where(non_origin_mask)[0]
            x_nonzero = x[non_origin_mask]
            y_nonzero = y[non_origin_mask]
            
            r = torch.sqrt(x_nonzero**2 + y_nonzero**2)
            phi_ = torch.atan2(y_nonzero, x_nonzero)
            
            # 将角度归一化到[0, 2π)
            phi = torch.where(phi_ < 0, phi_ + 2 * math.pi, phi_)
            
            if self.K == 1:
                # 计算反向角度（直接收敛区域）
                theta_ray = self.thetas[0]  # 对于K=1，取第一个角度
                reverse_theta = (theta_ray + math.pi) % (2 * math.pi)
                
                transition_width = math.pi / 8
                low_angle = reverse_theta - transition_width
                high_angle = reverse_theta + transition_width
                
                # 计算角度差（考虑周期性）
                diff_to_reverse = torch.abs(phi - reverse_theta)
                diff_to_reverse = torch.min(diff_to_reverse, 2 * math.pi - diff_to_reverse)
                
                # 确定区域类型
                direct_region_mask = diff_to_reverse <= transition_width
                transition_region_mask = (diff_to_reverse > transition_width) & (diff_to_reverse < 2 * transition_width)
                ray_region_mask = diff_to_reverse >= 2 * transition_width

                # 直接收敛区域 - 使用解析解
                if direct_region_mask.any():
                    # 获取直接收敛区域的索引
                    direct_indices = non_origin_indices[direct_region_mask]
                    # 直接收敛区域的速度场：v = -a * [x, y]
                    v_direct = -self.a * torch.stack([x_nonzero[direct_region_mask], 
                                                    y_nonzero[direct_region_mask]], dim=1)
                    velocity[direct_indices] = v_direct
                
                # 射线收敛区域 - 使用解析解
                if ray_region_mask.any():
                    # 获取射线收敛区域的索引
                    ray_indices = non_origin_indices[ray_region_mask]
                    # 计算角度差
                    delta_phi = phi_[ray_region_mask] - theta_ray
                    delta_phi = (delta_phi + math.pi) % (2 * math.pi) - math.pi
                    
                    # 射线收敛区域的速度场解析解
                    x_ray = x_nonzero[ray_region_mask]
                    y_ray = y_nonzero[ray_region_mask]
                    r_ray = r[ray_region_mask]
                    
                    # 计算速度分量
                    vx = -self.a * x_ray - self.b * (x_ray * delta_phi**2 - y_ray * delta_phi)
                    vy = -self.a * y_ray - self.b * (y_ray * delta_phi**2 + x_ray * delta_phi)
                    
                    v_ray = torch.stack([vx, vy], dim=1)
                    velocity[ray_indices] = v_ray
                
                # 过渡区域 - 使用自动求导
                if transition_region_mask.any():
                    # 获取过渡区域的索引
                    transition_indices = non_origin_indices[transition_region_mask]
                    xy_transition = xy_tensor[transition_indices].clone().requires_grad_(True)
                    U_transition = self.compute_potential(xy_transition)
                    
                    gradients = torch.autograd.grad(
                        U_transition, xy_transition, 
                        grad_outputs=torch.ones_like(U_transition),
                        create_graph=True,
                        retain_graph=True
                    )[0]
                    
                    v_transition = -gradients.detach()
                    velocity[transition_indices] = v_transition

            else:
                # 对于K>1的情况，计算所有射线角度和中间角度
                thetas_rays = self.thetas

                # thetas_middle = torch.tensor([(2 * i + 1) * math.pi / self.K for i in range(self.K)], dtype=xy_tensor.dtype, device=xy_tensor.device)
                # 对射线角度排序并考虑周期性
                sorted_thetas = torch.sort(self.thetas).values
                # 计算相邻射线之间的中间角度（包括首尾之间的）
                thetas_middle = (sorted_thetas + torch.roll(sorted_thetas, -1)) / 2
                # 处理首尾之间的中间角度
                thetas_middle[-1] = (sorted_thetas[-1] + sorted_thetas[0] + 2 * math.pi) / 2
                thetas_middle = thetas_middle % (2 * math.pi)

                # 计算与所有中间角度的最小差值
                phi_expanded = phi.unsqueeze(1)  # [batch_nonzero, 1]
                diffs = torch.abs(phi_expanded - thetas_middle)
                diffs = torch.min(diffs, 2 * math.pi - diffs)
                min_diff_m, min_middle_indices = torch.min(diffs, dim=1)
                
                # 计算与所有射线角度的最小差值并找到最近的射线
                diffs = torch.abs(phi_expanded - thetas_rays)
                diffs = torch.min(diffs, 2 * math.pi - diffs)
                min_diff_r, min_indices = torch.min(diffs, dim=1)
                theta_r_star = thetas_rays[min_indices]
                
                # # 确定区域类型
                # inner_angle = math.pi / (4 * self.K)
                # outer_angle = math.pi / (2 * self.K)

                # 计算相邻射线之间的角度间隔
                angles_diff = torch.diff(sorted_thetas, append=sorted_thetas[0].unsqueeze(0) + 2 * math.pi)
                angles_diff = angles_diff % (2 * math.pi)

                point_intervals = angles_diff[min_middle_indices]

                # 基于实际间隔计算阈值
                inner_angle = point_intervals / 10
                outer_angle = point_intervals / 5
                
                direct_region_mask = min_diff_m <= inner_angle
                transition_region_mask = (min_diff_m > inner_angle) & (min_diff_m < outer_angle)
                ray_region_mask = min_diff_m >= outer_angle

                region_type[transition_region_mask] = 1
                region_type[ray_region_mask] = 2
                
                # 直接收敛区域 - 使用解析解
                if direct_region_mask.any():
                    # 获取直接收敛区域的索引
                    direct_indices = non_origin_indices[direct_region_mask]
                    v_direct = -self.a * torch.stack([x_nonzero[direct_region_mask], 
                                                    y_nonzero[direct_region_mask]], dim=1)
                    velocity[direct_indices] = v_direct
                
                # 射线收敛区域 - 使用解析解
                if ray_region_mask.any():
                    # 获取射线收敛区域的索引
                    ray_indices = non_origin_indices[ray_region_mask]
                    # 计算角度差
                    delta_phi = phi[ray_region_mask] - theta_r_star[ray_region_mask]
                    delta_phi = (delta_phi + math.pi) % (2 * math.pi) - math.pi
                    
                    # 射线收敛区域的速度场解析解
                    x_ray = x_nonzero[ray_region_mask]
                    y_ray = y_nonzero[ray_region_mask]
                    
                    # 计算速度分量
                    vx = -self.a * x_ray - self.b * (x_ray * delta_phi**2 - y_ray * delta_phi)
                    vy = -self.a * y_ray - self.b * (y_ray * delta_phi**2 + x_ray * delta_phi)
                    
                    v_ray = torch.stack([vx, vy], dim=1)
                    velocity[ray_indices] = v_ray
                
                # 过渡区域 - 使用自动求导
                if transition_region_mask.any():
                    # 获取过渡区域的索引
                    transition_indices = non_origin_indices[transition_region_mask]
                    xy_transition = xy_tensor[transition_indices].clone().requires_grad_(True)
                    U_transition = self.compute_potential(xy_transition)
                    
                    gradients = torch.autograd.grad(
                        U_transition, xy_transition, 
                        grad_outputs=torch.ones_like(U_transition),
                        create_graph=True,
                        retain_graph=True
                    )[0]
                    
                    v_transition = -gradients.detach()
                    velocity[transition_indices] = v_transition
        
        return velocity, region_type

if __name__ == "__main__":
    # 示例用法
    vf = KCanyon_Combined_Field()

