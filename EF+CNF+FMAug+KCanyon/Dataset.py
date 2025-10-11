from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from Canyon_models import MLP
from EnvelopRegion import TrajectoryRegionClassifier
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def generate_global_augment_dataset(file_name):
    # 首先使用训练好的OTCFM模型来生成增强的训练数据

    positions, velocities, lasa_indices = ImportLASADataset(f'{file_name}.mat')
    x_min, x_max = positions[:, 0].min() - 20, positions[:, 0].max() + 20
    y_min, y_max = positions[:, 1].min() - 20, positions[:, 1].max() + 20
    limit=np.array([[x_min,y_min],[x_max,y_max]])

    model = MLP(dim=2, time_varying=False).to(device)
    model.load_state_dict(torch.load(f'models/OTCFM_x/OTCFM_x_{file_name}.pth'))
    # gmm=joblib.load(f'models/GMM/trajectory_gmm_{file_name}.pkl')
    r1 = 2  # 内部区域半径
    r2 = 5   # 边界区域外半径
    regionClassifier = TrajectoryRegionClassifier(positions, r1, r2)

    # OTCFM生成的数据与示教数据合并
    Nt=100
    Np=1024
    test_points=regionClassifier.sample_source_points(Np, limit).to(device)
    res_xt, res_vt=OTCFM_x_inference(model,test_points,Nt,Np)

    dataset = GlobalTrajectoryDataset(regionClassifier, res_xt, res_vt, positions, velocities, r1, r2, file_name)
    loader = DataLoader(dataset, batch_size=512, shuffle=True)
    return dataset, loader, positions/dataset.scale, limit/dataset.scale


class GlobalTrajectoryDataset(Dataset):
    def __init__(self, classifier, fm_x, fm_v, lasa_x, lasa_v, r1, r2, file_name):
        self.file_name=file_name
        self.r1=r1
        self.r2=r2
        self.classifier = classifier
        
        self.fm_x = fm_x.reshape(-1, 2)
        self.fm_v = fm_v.reshape(-1, 2)
        self.lasa_x = lasa_x
        self.lasa_v = lasa_v
        self.smooth()

        # # 将lasa数据和fm增强数据合并
        x_np = np.concatenate((self.fm_x, self.lasa_x), axis=0)
        v_np = np.concatenate((self.fm_v, self.lasa_v), axis=0)

        # # 只使用fm增强数据
        # x_np = self.fm_x
        # v_np = self.fm_v

        target = np.array([0, 0])
        demonstrations={
            'positions': x_np,
            'velocities': v_np,
            'target': target
        }

        # 注意,应当先对原始lasa数据进行归一化,然后算出scale,再用同样的scale将增强数据进行缩放,保持和之前训练数据同样的scale
        lasa_demos={
            'positions': lasa_x,
            'velocities': lasa_v,
            'target': target
        }
        scale = self.compute_scale(lasa_demos)
        max_magnitude=self.compute_maxscale(lasa_demos)
        # 使用同样的scale将增强数据进行缩放
        normalized_demos, _ = self.prepare_demonstrations_scale(demonstrations, scale)
        positions, velocities, targets = self.create_training_data(normalized_demos)

        self.x = positions
        self.v = velocities
        self.targets = targets
        self.scale=scale
        self.max_magnitude=max_magnitude

        # self.SaveFMAugDataFig(positions, velocities)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.v[idx], self.targets[idx]
    
    def prepare_demonstrations_scale(self,demo,scale):
        # 获取位置数据
        positions = demo['positions']
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
    
    def compute_scale(self,demo):
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
        return scale
    
    def compute_maxscale(self, demo):

        positions = demo['positions']

        norms = np.linalg.norm(positions, axis=1)
    
        # Find the maximum magnitude
        max_magnitude = np.max(norms)
        
        # Avoid division by zero (safety check)
        max_magnitude = max(max_magnitude, 1e-12)
        return max_magnitude


    def create_training_data(self, demo):
        """
        从归一化的演示创建训练数据
        """
        # positions = []
        # velocities = []
        # targets = []
        
        # positions.append()
        # velocities.append(demo['velocities'])
        # targets.extend([demo['target']] * len(demo['positions']))
        
        positions = demo['positions']
        velocities = demo['velocities']
        targets = np.repeat([demo['target']], len(demo['positions']), axis=0)

        return torch.FloatTensor(positions).to(device), torch.FloatTensor(velocities).to(device), torch.FloatTensor(targets).to(device)

    def smooth(self,):
        # 增强数据和示教数据在交界处存在突变，导致在边界处的学习效果不太好
        # 考虑将边界附近的增强数据速度进行平滑优化
        # 设定一个边界范围，例如a<NLL<b的区间，或者包络圆半径r1<r<r2的区间
        # 对增强数据中的每个数据点做以下操作
        # 如果该数据点在边界范围内，计算该点的NLL，并查找其最近的一个示教点，然后将其速度该为加权和

        boundary_fm_x, boundary_fm_v, boundary_dists, boundary_indices = self.boundary_select(self.r1, self.r2)
        # 保存边界点和原始速度
        self.boundary_indices = boundary_indices
        self.boundary_original_v = boundary_fm_v.copy()
        self.update_v(self.lasa_x, self.lasa_v, boundary_fm_x, boundary_fm_v, boundary_indices, boundary_dists, self.r1, self.r2)

    def boundary_select(self,r1,r2):
        inner_mask, boundary_mask, outer_mask, dists = self.classifier.classify_points(self.fm_x)
        self.fm_x = self.fm_x[dists > r1]
        self.fm_v = self.fm_v[dists > r1]

        boundary_indices, boundary_fm_x, boundary_dists = self.classifier.get_points_in_region(self.fm_x, 'boundary', return_dists=True)
        boundary_fm_v = self.fm_v[boundary_indices]

        return boundary_fm_x, boundary_fm_v, boundary_dists, boundary_indices

    def update_v(self, points1, velocities1, points2, velocities2, mask, r_values, r1, r2):
        """
        更新第二组点的速度
        
        参数:
            points1: 第一组点位置, 形状 [1000, 2] (NumPy数组)
            velocities1: 第一组点速度, 形状 [1000, 2] (NumPy数组)
            points2: 第二组点位置, 形状 [500, 2] (NumPy数组)
            velocities2: 第二组点速度, 形状 [500, 2] (NumPy数组)
            mask: 选择第二组点的布尔掩码, 形状 [500]
            r_values: 第二组点的r值, 形状 [500] (NumPy数组)
            r1, r2: r值的范围边界 (标量)
        
        返回:
            更新后的第二组点速度, 形状 [500, 2] (NumPy数组)
        """
        # 步骤1: 计算所有点对之间的欧氏距离矩阵
        # 使用广播机制计算 (500,1,2) - (1000,2) -> (500,1000,2)
        diff = points2[:, np.newaxis, :] - points1[np.newaxis, :, :]
        
        # 计算平方距离 (500,1000)
        dist_sq = np.sum(diff**2, axis=2)
        
        # 步骤2: 为每个points2点找到最近的points1点索引
        # 找到最小距离的索引 (500)
        nearest_indices = np.argmin(dist_sq, axis=1)
        
        # 步骤3: 获取最近点的速度
        nearest_velocities = velocities1[nearest_indices]  # 形状 [500, 2]
        
        # 步骤4: 计算混合系数beta
        beta = (r_values - r1) / (r2 - r1)  # 形状 [500]
        beta = beta[:, np.newaxis]  # 扩展为 [500, 1] 以便广播
        
        # 步骤5: 计算新的速度
        new_velocities = beta * velocities2 + (1 - beta) * nearest_velocities
        
        # 步骤6: 更新指定点的速度
        self.fm_v[mask] = new_velocities
    
    def SaveFMAugDataFig(self, positions, velocities):
        """
        使用流线图可视化速度场
        """
        if hasattr(positions, 'detach'):
            positions = positions.detach().cpu().numpy()
        if hasattr(velocities, 'detach'):
            velocities = velocities.detach().cpu().numpy()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 创建网格
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
        
        # 扩展边界
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        
        x_grid = np.linspace(x_min - x_padding, x_max + x_padding, 50)
        y_grid = np.linspace(y_min - y_padding, y_max + y_padding, 50)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # 插值速度场到网格（简单最近邻插值）
        from scipy.interpolate import griddata
        U = griddata(positions, velocities[:, 0], (X, Y), method='nearest')
        V = griddata(positions, velocities[:, 1], (X, Y), method='nearest')
        
        # 绘制流线图
        speed = np.sqrt(U**2 + V**2)
        # strm = ax.streamplot(X, Y, U, V, density=2, color=speed, cmap='viridis', linewidth=1.5)
        strm = ax.streamplot(X, Y, U, V, density=2, color='black', cmap='viridis', linewidth=1.5)
        
        # # 绘制位置点
        # ax.scatter(positions[:, 0], positions[:, 1], c='red', s=30, alpha=0.7)
        
        # 添加颜色条
        # fig.colorbar(strm.lines, ax=ax, label='Velocity Magnitude')
        plt.tight_layout()
        plt.xticks([])
        plt.yticks([])
        # plt.show()
        plt.savefig(f'visualizations/FMAugDataFig/FM_{self.file_name}.png', dpi=300)


class FMDiffeoDataset(Dataset):
    def __init__(self, x, v, target):
        x = x.reshape(-1, 2)
        v = v.reshape(-1, 2)
        self.x = x.to(device) 
        self.v = v.to(device)
        self.target = target.to(device)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.v[idx], self.target[idx]