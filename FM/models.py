import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, dim, out_dim=None, w=64, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        # self.net = torch.nn.Sequential(
        #     torch.nn.Linear(dim + (1 if time_varying else 0), w),
        #     torch.nn.SELU(),
        #     torch.nn.Linear(w, w),
        #     torch.nn.SELU(),
        #     torch.nn.Linear(w, w),
        #     torch.nn.SELU(),
        #     torch.nn.Linear(w, out_dim),
        # )
        self.net = nn.Sequential(
            nn.Linear(dim + (1 if time_varying else 0), w),
            nn.Softplus(),
            nn.Linear(w, w),
            nn.Softplus(),
            nn.Linear(w, w),
            nn.Softplus(),
            nn.Linear(w, out_dim),
        )
        
    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        return self.net(x)
    
class MLP_x(nn.Module):
    def __init__(self, dim, out_dim=None, w=128, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        # self.net = torch.nn.Sequential(
        #     torch.nn.Linear(dim + (1 if time_varying else 0), w),
        #     torch.nn.SELU(),
        #     torch.nn.Linear(w, w),
        #     torch.nn.SELU(),
        #     torch.nn.Linear(w, w),
        #     torch.nn.SELU(),
        #     torch.nn.Linear(w, out_dim),
        # )
        self.net = nn.Sequential(
            nn.Linear(dim + (1 if time_varying else 0), w),
            nn.Softplus(),
            nn.Linear(w, w),
            nn.Softplus(),
            nn.Linear(w, w),
            nn.Softplus(),
            nn.Linear(w, w),
            nn.Softplus(),
            nn.Linear(w, out_dim),
        )
        
    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        return self.net(x)


class StableMLP_V1(nn.Module):
    def __init__(self, w=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, w),
            nn.Tanh(),
            nn.Linear(w, w),
            nn.Tanh(),
            nn.Linear(w, w),
            nn.Tanh(),
            nn.Linear(w, 1),
        )
        # 使用绝对值参数化保证最后一层权重非负
        self._parametrize_last_layer()
        
    def _parametrize_last_layer(self):
        last_linear = self.net[-1]
        last_linear.weight = nn.Parameter(torch.abs(last_linear.weight.data))
        last_linear.weight.register_hook(lambda grad: torch.abs(grad))

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x.requires_grad_(True)
        
        # 计算网络输出
        s = self.net(x)
        
        # 计算输出对输入的梯度（负梯度）
        grad_s = torch.autograd.grad(
            s, x, 
            grad_outputs=torch.ones_like(s),
            create_graph=True,
            retain_graph=True
        )[0]

        return -grad_s

class StableMLP_V2(nn.Module):
    def __init__(self, w=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, w),
            nn.Tanh(),
            nn.Linear(w, w),
            nn.Tanh(),
            nn.Linear(w, w),
            nn.Tanh(),
            nn.Linear(w, 1),
        )
        # 添加额外的g网络
        self.g = nn.Sequential(
            nn.Linear(2, w),
            nn.Tanh(),
            nn.Linear(w, w),
            nn.Tanh(),
            nn.Linear(w, w),
            nn.Tanh(),
            nn.Linear(w, 2),
        )
        # 使用绝对值参数化保证最后一层权重非负
        self._parametrize_last_layer()
        
    def _parametrize_last_layer(self):
        last_linear = self.net[-1]
        last_linear.weight = nn.Parameter(torch.abs(last_linear.weight.data))
        last_linear.weight.register_hook(lambda grad: torch.abs(grad))

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x.requires_grad_(True)
        
        # 计算s网络输出
        s = self.net(x)
        
        # 计算s对输入的梯度
        grad_s = torch.autograd.grad(
            s, x, 
            grad_outputs=torch.ones_like(s),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # 计算g网络输出
        g_out = self.g(x)
        
        # gout-(g_out在grad_s方向上的投影)
        # v1
        dot_product = (g_out * grad_s).sum(dim=1, keepdim=True)
        grad_s_norm_sq = (grad_s ** 2).sum(dim=1, keepdim=True) + 1e-8  # 避免除零
        proj = (dot_product / grad_s_norm_sq) * grad_s
        f=g_out - proj

        # # v2
        # # 剔除g(x)沿梯度dVdx的正分量
        # # 计算g沿dVdx方向的分量（投影）
        # norm_sq = torch.sum(grad_s * grad_s, dim=1, keepdim=True)           # 计算||dVdx||^2
        # dot_product = torch.sum(g_out * grad_s, dim=1, keepdim=True)          # 点积
        # # 使用逐元素操作
        # f = torch.where(norm_sq > 1e-8, g_out - (F.softplus(dot_product / norm_sq) + 1) * grad_s, g_out)
        
        return f
    

class StableMLP_V3(nn.Module):
    def __init__(self, w=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, w),
            nn.Tanh(),
            nn.Linear(w, w),
            nn.Tanh(),
            nn.Linear(w, w),
            nn.Tanh(),
            nn.Linear(w, 1),
        )
        # 添加额外的g网络
        self.g = nn.Sequential(
            nn.Linear(2, w),
            nn.Tanh(),
            nn.Linear(w, w),
            nn.Tanh(),
            nn.Linear(w, w),
            nn.Tanh(),
            nn.Linear(w, 2),
        )
        # 使用绝对值参数化保证最后一层权重非负
        self._parametrize_last_layer()
        
    def _parametrize_last_layer(self):
        last_linear = self.net[-1]
        last_linear.weight = nn.Parameter(torch.abs(last_linear.weight.data))
        last_linear.weight.register_hook(lambda grad: torch.abs(grad))

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x.requires_grad_(True)
        
        # 计算s网络输出
        s = self.net(x)
        
        # 计算s对输入的梯度
        grad_s = torch.autograd.grad(
            s, x, 
            grad_outputs=torch.ones_like(s),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # 计算g网络输出
        g_out = self.g(x)
        
        # gout-(g_out在grad_s方向上的投影)
        # v1
        # dot_product = (g_out * grad_s).sum(dim=1, keepdim=True)
        # grad_s_norm_sq = (grad_s ** 2).sum(dim=1, keepdim=True) + 1e-8  # 避免除零
        # proj = (dot_product / grad_s_norm_sq) * grad_s
        # f=g_out - proj

        # v2
        # 剔除g(x)沿梯度dVdx的正分量
        # 计算g沿dVdx方向的分量（投影）
        norm_sq = torch.sum(grad_s * grad_s, dim=1, keepdim=True)           # 计算||dVdx||^2
        dot_product = torch.sum(g_out * grad_s, dim=1, keepdim=True)          # 点积
        # 使用逐元素操作
        f = torch.where(norm_sq > 1e-8, g_out - (F.softplus(dot_product / norm_sq) + 1) * grad_s, g_out)
        return f