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
import time
        
warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

class VelocityField(nn.Module):
    """
    参数化速度场 u_t(x) 的神经网络
    """
    def __init__(self, dim, hidden_dim=256, num_layers=5):
        super(VelocityField, self).__init__()
        self.dim = dim
        
        # 时间相关的速度场
        layers = []
        layers.append(nn.Linear(dim + 1, hidden_dim))  # +1 用于时间维度
        layers.append(nn.Tanh())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(hidden_dim, dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, t, x):
        # 将时间与空间坐标连接
        t_tensor = torch.ones(x.shape[0], 1).to(x.device) * t
        x_with_time = torch.cat([x, t_tensor], dim=1)
        
        return self.net(x_with_time)

class FlowMatchingDiffeo(nn.Module):
    """
    通过流匹配学习的微分同胚
    """
    def __init__(self, dim, hidden_dim=256, num_layers=5):
        super(FlowMatchingDiffeo, self).__init__()
        self.dim = dim
        self.velocity_field = VelocityField(dim, hidden_dim, num_layers)
        
    def forward_ode(self, t, x_flat):
        """
        solve_ivp 的 ODE 函数
        """
        x = x_flat.reshape(-1, self.dim)
        x_tensor = torch.FloatTensor(x).to(self.device)
        
        # 移除 torch.no_grad() 以保留梯度
        dx_dt = self.velocity_field(t, x_tensor).cpu().detach().numpy()
            
        return dx_dt.flatten()
    
    def inverse_ode(self, t, x_flat):
        """
        反向变换的逆 ODE 函数
        """
        return -self.forward_ode(1-t, x_flat)
    
    def forward(self, x, method='RK45', rtol=1e-7, atol=1e-9):
        """
        应用微分同胚变换: x -> y
        """
        self.device = x.device
        x_np = x.detach().cpu().numpy()
        
        # 从 t=0 到 t=1 求解 ODE
        solution = solve_ivp(
            self.forward_ode,
            [0, 1],
            x_np.flatten(),
            method=method,
            rtol=rtol,
            atol=atol
        )
        
        y_np = solution.y[:, -1].reshape(x_np.shape)
        return torch.FloatTensor(y_np).to(self.device).requires_grad_(True)
    
    def inverse(self, y, method='RK45', rtol=1e-7, atol=1e-9):
        """
        应用逆变换: y -> x
        """
        self.device = y.device
        y_np = y.detach().cpu().numpy()
        
        # 从 t=0 到 t=1 求解反向 ODE（相当于从 t=1 到 t=0 的正向求解）
        solution = solve_ivp(
            self.inverse_ode,
            [0, 1],
            y_np.flatten(),
            method=method,
            rtol=rtol,
            atol=atol
        )
        
        x_np = solution.y[:, -1].reshape(y_np.shape)
        return torch.FloatTensor(x_np).to(self.device).requires_grad_(True)
    
    def compute_velocity(self, x, t):
        """
        计算位置 x 和时间 t 处的速度
        """
        return self.velocity_field(t, x)
    
    def differentiable_forward(self, x):
        """
        可微分的正向变换实现，使用欧拉方法
        这是一个替代方案，避免了使用 scipy.integrate.solve_ivp
        """
        
        y = x.clone()
        
        # # 使用欧拉方法进行数值积分
        # n_steps=100
        # dt = 1.0 / n_steps
        # for i in range(n_steps):
        #     t = i * dt
        #     velocity = self.velocity_field(t, y)
        #     y = y + velocity * dt
        
        # 使用四阶龙格-库塔方法 (RK4)
        step = 10
        dt = 1/step
        for i in range(step):
            t = i * dt
            k1 = self.velocity_field(t, y)
            k2 = self.velocity_field(t + dt/2, y + dt/2 * k1)
            k3 = self.velocity_field(t + dt/2, y + dt/2 * k2)
            k4 = self.velocity_field(t + dt, y + dt * k3)
            y = y + (k1 + 2*k2 + 2*k3 + k4) * dt / 6
            
        return y
    
    def differentiable_inverse(self, y):
        """
        可微分的逆向变换实现，使用欧拉方法
        这是一个替代方案，避免了使用 scipy.integrate.solve_ivp
        """
        x = y.clone()

        # # 使用欧拉方法进行反向数值积分
        # n_steps=100
        # dt = 1.0 / n_steps
        # for i in range(n_steps, 0, -1):
        #     t = i * dt
        #     velocity = self.velocity_field(t, x)
        #     x = x - velocity * dt

        # 使用四阶龙格-库塔方法 (RK4)
        step = 10
        dt = 1/step
        for i in range(step, 0, -1):
            t = i * dt
            # RK4反向积分
            k1 = self.velocity_field(t, x)
            k2 = self.velocity_field(t - dt/2, x - dt/2 * k1)
            k3 = self.velocity_field(t - dt/2, x - dt/2 * k2)
            k4 = self.velocity_field(t - dt, x - dt * k3)
            x = x - (k1 + 2*k2 + 2*k3 + k4) * dt / 6
        return x

class SDSEF_FM(nn.Module):
    """
    使用流匹配的欧几里得化流的稳定动力系统
    """
    def __init__(self, dim, hidden_dim=128, num_layers=4):
        super(SDSEF_FM, self).__init__()
        self.dim = dim
        self.diffeo = FlowMatchingDiffeo(dim, hidden_dim, num_layers)
        
    def forward(self, x, x_star):
        """
        计算给定目标 x_star 的位置 x 处的速度
        """
        # 启用梯度计算以进行雅可比计算
        x.requires_grad_(True)
        
        combined = torch.cat([x, x_star], dim=0)  # 形状为 [n+1, 2]
        y_combined = self.diffeo.differentiable_forward(combined)
        y = y_combined[:x.size(0)]
        y_star = y_combined[x.size(0):]

        # 计算潜在空间中的势函数梯度
        # Φ(y) = ||y - y*||
        diff = y - y_star
        norm = torch.norm(diff, dim=1, keepdim=True) + 1e-8  # 避免除以零
        grad_phi_y = diff / norm

        # 使用自动微分计算微分同胚的雅可比矩阵
        jacobian = torch.zeros(x.size(0), self.dim, self.dim).to(x.device)
        for i in range(self.dim):
            grad = torch.autograd.grad(
                y[:, i].sum(), x, create_graph=True, retain_graph=True
            )[0]
            jacobian[:, i, :] = grad
        
        # 计算原始空间中的自然梯度下降
        # ẋ = -G_ψ(x)^{-1} ∇_x(Φ∘ψ)(x)
        # 其中 G_ψ(x) = J_ψ(x)^T J_ψ(x)
        G = torch.matmul(jacobian.transpose(1, 2), jacobian)
        G_inv = torch.inverse(G + 1e-6 * torch.eye(self.dim).unsqueeze(0).to(x.device))
        
        # 计算 ∇_x(Φ∘ψ)(x) = J_ψ(x)^T ∇_yΦ(y)
        grad_phi_x = torch.matmul(jacobian.transpose(1, 2), grad_phi_y.unsqueeze(2)).squeeze(2)
        
        # 计算速度
        x_dot = -torch.matmul(G_inv, grad_phi_x.unsqueeze(2)).squeeze(2)

        return x_dot
    
    def direct_velocity_prediction(self, x, x_star):
        """
        替代方法：使用流匹配框架直接预测速度
        这可能更高效，但可能不保持相同的稳定性保证
        """
        # 映射到潜在空间 - 使用可微分版本
        combined = torch.cat([x, x_star], dim=0)  # 形状为 [n+1, 2]
        y_combined = self.diffeo.differentiable_forward(combined)
        y = y_combined[:x.size(0)]
        y_star = y_combined[x.size(0):]
        
        # 计算潜在空间中的直线速度
        diff = y - y_star
        norm = torch.norm(diff, dim=1, keepdim=True) + 1e-8
        latent_velocity = -diff / norm

        # 使用有限差分近似雅可比矩阵
        epsilon = 1e-4
        
        # 预先分配内存
        batch_size, dim = x.shape
        jacobian_approx = torch.zeros(batch_size, dim, dim, device=x.device)
        perturbations = torch.zeros(2 * dim * batch_size, dim, device=x.device)
        
        # 创建所有扰动输入
        for i in range(dim):
            # 正扰动
            start_idx = 2 * i * batch_size
            end_idx = (2 * i + 1) * batch_size
            perturbations[start_idx:end_idx] = x
            perturbations[start_idx:end_idx, i] += epsilon
            
            # 负扰动
            start_idx = (2 * i + 1) * batch_size
            end_idx = (2 * i + 2) * batch_size
            perturbations[start_idx:end_idx] = x
            perturbations[start_idx:end_idx, i] -= epsilon

        # 一次性计算所有扰动的输出
        outputs_batch = self.diffeo.differentiable_forward(perturbations)
        # 拆分结果并计算雅可比矩阵
        for i in range(dim):
            # 获取正扰动和负扰动的输出
            y_plus = outputs_batch[2*i*batch_size : (2*i+1)*batch_size]
            y_minus = outputs_batch[(2*i+1)*batch_size : (2*i+2)*batch_size]
            # 计算有限差分
            jacobian_approx[:, :, i] = (y_plus - y_minus) / (2 * epsilon)


        # 使用伪逆而不是精确逆
        jacobian_pinv = torch.pinverse(jacobian_approx)
        
        # 将速度变换回原始空间
        x_dot = torch.bmm(jacobian_pinv, latent_velocity.unsqueeze(2)).squeeze(2)
        
        return x_dot
    
    def inference(self, x, x_star, dt=0.01, steps=2000):
        traj=[]
        traj.append(x)
        # for i in range(steps):
        i=0
        while(i<steps):
            x = x + dt * self.direct_velocity_prediction(x, x_star)
            traj.append(x)
            i += 1
            if torch.norm(x - x_star) < 1e-3:
                break
        traj_np = np.stack([t.detach().cpu().numpy() if hasattr(t, 'cpu') else np.array(t) for t in traj])
        return traj_np
    
    def uniform_interpolation(self, x0, x_star):
        # 先将x和x*映射到y空间
        combined = torch.cat([x0, x_star], dim=0)  # 形状为 [n+1, 2]
        y_combined = self.diffeo.differentiable_forward(combined)
        y = y_combined[:x0.size(0)]
        y_star = y_combined[x0.size(0):]
        # 起始点到终点间设定1000步
        Ka=1000
        self.distance_y = torch.norm(y_star - y)
        self.interval=self.distance_y/(Ka-1)
        return self.interval

    def chunkpredict(self,x, x_star):
        
        combined = torch.cat([x, x_star], dim=0)  # 形状为 [n+1, 2]
        y_combined = self.diffeo.differentiable_forward(combined)
        y = y_combined[:x.size(0)]
        y_star = y_combined[x.size(0):]

        # 在y到y*的直线路径上均匀插值K个，起点是y，终点是y*
        distance = torch.norm(y_star - y)
        K = int(distance / self.interval)
        # 生成插值参数t，从0到1均匀分布
        t_values = torch.linspace(0, 1, K, device=y.device)
        # 线性插值公式: point = start + t * (end - start)
        interpolated_y = y + t_values.unsqueeze(1) * (y_star - y)

        # 将插值点映射回x空间
        x_traj = self.diffeo.differentiable_inverse(interpolated_y)

        # print('yes')
        return x_traj