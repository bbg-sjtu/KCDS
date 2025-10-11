from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FMTrajectoryDataset(Dataset):
    def __init__(self, x, v):
        x = x.reshape(-1, 2)
        v = v.reshape(-1, 2)
        self.x = torch.from_numpy(x).to(device) 
        self.v = torch.from_numpy(v).to(device)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.v[idx]
    
class InnerTrajectoryDataset(Dataset):
    def __init__(self, positions, velocities, lasa_indices):
        positions = positions.reshape(-1, 2)
        velocities = velocities.reshape(-1, 2)
        self.x = torch.from_numpy(positions).to(device) 
        self.v = torch.from_numpy(velocities).to(device)
        self.indices = torch.from_numpy(lasa_indices).to(device)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.v[idx], self.indices[idx]

class GlobalTrajectoryDataset(Dataset):
    def __init__(self, classifier, fm_x, fm_v, lasa_x, lasa_v, lasa_indices, r1, r2):
        self.r1=r1
        self.r2=r2
        self.classifier = classifier
        
        self.fm_x = fm_x.reshape(-1, 2)
        self.fm_v = fm_v.reshape(-1, 2)
        self.lasa_x = lasa_x
        self.lasa_v = lasa_v
        self.smooth()

        lasa_indices = torch.from_numpy(lasa_indices).to(device)  # 确保lasa_indices也是tensor
        fm_indices = np.zeros(self.fm_x.shape[0], dtype=int)  # 形状与fm_x相同，全0
        fm_indices = torch.from_numpy(fm_indices).to(device)  # 转换为tensor
 
        self.fm_x=torch.from_numpy(self.fm_x).to(device)
        self.fm_v=torch.from_numpy(self.fm_v).to(device)
        self.lasa_x=torch.from_numpy(self.lasa_x).to(device)
        self.lasa_v=torch.from_numpy(self.lasa_v).to(device)

        self.x = torch.cat((self.fm_x, self.lasa_x), dim=0)
        self.v = torch.cat((self.fm_v, self.lasa_v), dim=0)
        self.indices = torch.cat((fm_indices, lasa_indices), dim=0)  # 合并indices
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.v[idx], self.indices[idx]

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
    
    def plot_velocity_field(self,):
        """可视化速度场，特别展示加权平滑效果"""
        plt.figure(figsize=(14, 12))
        
        # 转换回NumPy用于可视化
        fm_x = self.fm_x.cpu().numpy() if torch.is_tensor(self.fm_x) else self.fm_x
        fm_v = self.fm_v.cpu().numpy() if torch.is_tensor(self.fm_v) else self.fm_v
        lasa_x = self.lasa_x.cpu().numpy() if torch.is_tensor(self.lasa_x) else self.lasa_x
        lasa_v = self.lasa_v.cpu().numpy() if torch.is_tensor(self.lasa_v) else self.lasa_v
        
        # 1. 绘制所有增强数据点（FM）
        plt.scatter(fm_x[:, 0], fm_x[:, 1], c='blue', alpha=0.4, s=10, label='FM Points')
        
        # 2. 绘制边界点（特殊标记）
        if self.boundary_indices is not None:
            boundary_x = fm_x[self.boundary_indices]
            plt.scatter(boundary_x[:, 0], boundary_x[:, 1], c='red', alpha=0.7, s=40, 
                        marker='s', label='Boundary Points')
        
        # 3. 绘制示教数据点（LASA）
        plt.scatter(lasa_x[:, 0], lasa_x[:, 1], c='green', alpha=0.7, s=60, 
                    marker='^', label='LASA Points')
        
        # 4. 绘制速度向量
        scale = 0.001  # 箭头缩放比例
        plt.quiver(fm_x[:, 0], fm_x[:, 1],
                    fm_v[:, 0], fm_v[:, 1],
                    scale=1/scale, color='blue', width=0.003, alpha=0.4, label='FM Velocity')

        plt.title("Velocity Field")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        # # 添加颜色条表示r值
        # if self.boundary_mask is not None:
        #     boundary_nll = -self.gmm.score_samples(boundary_x)
        #     sc = plt.scatter(boundary_x[:, 0], boundary_x[:, 1], c=boundary_nll, 
        #                      cmap='viridis', alpha=0.7, s=40, marker='s')
        #     cbar = plt.colorbar(sc, label='NLL (Negative Log-Likelihood)')
        #     cbar.set_label('Boundary Point NLL Value', fontsize=12)
        
        # if save_path:
        #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
        #     print(f"Saved velocity field plot to {save_path}")
        
        plt.show()


class EnhancedGlobalTrajectoryDataset(GlobalTrajectoryDataset):
    def __init__(self, classifier, fm_x, fm_v, lasa_x, lasa_v, lasa_indices, r1, r2, num_new_points=1000):
        """
        Args:
            r2: 新的外边界半径 (r2 > r1)
            num_new_points: 在终点附近生成的新数据点数量
        """
        super().__init__(classifier, fm_x, fm_v, lasa_x, lasa_v, lasa_indices, r1, r2)
        
        # 保存原始fm数据长度（用于后续indices重建）
        self.original_fm_len = len(self.fm_x)
        
        # 1. 计算r1处的平均速度
        va_magnitude_avg = self._compute_va_magnitude_avg(r1)
        
        # 2. 在r1圆内生成新数据点
        new_x, new_v = self._generate_new_points_in_circle(r1, num_new_points, va_magnitude_avg)
        
        # 3. 将新点添加到fm数据中
        self.fm_x = torch.cat([self.fm_x, new_x], dim=0)
        self.fm_v = torch.cat([self.fm_v, new_v], dim=0)
        # 4. 在r1<r<r2环形区域进行平滑
        self._smooth_ring_region(r1, r2, va_magnitude_avg)
        
        # 5. 重新合并所有数据
        self.x = torch.cat((self.fm_x, self.lasa_x), dim=0)
        self.v = torch.cat((self.fm_v, self.lasa_v), dim=0)
        
        # 6. 重建indices
        new_fm_indices = torch.zeros(len(self.fm_x), dtype=torch.long, device=self.x.device)
        original_lasa_indices = self.indices[self.original_fm_len:self.original_fm_len + len(self.lasa_x)]
        self.indices = torch.cat((new_fm_indices, original_lasa_indices), dim=0)
    
    def _compute_va_magnitude_avg(self, r1, epsilon=0.001):
        """计算7条轨迹在r1处的速度大小平均值"""
        magnitudes = []
        
        # 遍历7条轨迹
        for i in range(7):
            start_idx = i * 1000
            end_idx = (i + 1) * 1000
            traj_x = self.lasa_x[start_idx:end_idx]
            traj_v = self.lasa_v[start_idx:end_idx]
            
            # 计算到终点的距离
            dists = torch.norm(traj_x, dim=1)
            
            # 从轨迹末尾向前查找r1附近的点
            found = False
            for j in range(end_idx - 1, start_idx - 1, -1):  # 从末尾向前搜索
                local_idx = j - start_idx
                if dists[local_idx] >= r1 - epsilon and dists[local_idx] <= r1 + epsilon:
                    speed = torch.norm(traj_v[local_idx])
                    magnitudes.append(speed.item())
                    found = True
                    break
            
            # 如果没找到符合条件的点，使用最近的点
            if not found:
                # 找到最接近r1的点
                diff = torch.abs(dists - r1)
                closest_idx = torch.argmin(diff)
                speed = torch.norm(traj_v[closest_idx])
                magnitudes.append(speed.item())
        
        # 计算平均速度大小
        return sum(magnitudes) / len(magnitudes)
    
    def _generate_new_points_in_circle(self, r1, num_points, va_magnitude_avg):
        """在r1圆内生成新数据点"""
        device = self.x.device
        
        # 均匀采样（极坐标）
        r = torch.sqrt(torch.rand(num_points, device=device)) * r1  # sqrt确保均匀分布
        theta = torch.rand(num_points, device=device) * 2 * math.pi
        
        # 转换为笛卡尔坐标
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        new_x = torch.stack([x, y], dim=1)
        
        # 计算速度：大小与距离成正比，方向指向终点
        speed_scale = va_magnitude_avg / r1
        new_v = -speed_scale * new_x  # 负号表示指向原点
        
        return new_x, new_v
    
    def _smooth_ring_region(self, r1, r2, va_magnitude_avg):
        """在r1<r<r2环形区域进行速度平滑"""
        device = self.x.device
        speed_scale = va_magnitude_avg / r1

        # 1. 处理FM数据
        fm_dists = torch.norm(self.fm_x, dim=1)
        fm_mask = (fm_dists > r1) & (fm_dists < r2)
        
        if torch.any(fm_mask):
            # 计算目标速度（指向终点）
            target_v = -speed_scale * self.fm_x[fm_mask]
            
            # 计算混合系数
            m = (fm_dists[fm_mask] - r1) / (r2 - r1)
            m = m.unsqueeze(1)  # 扩展维度用于广播
            
            # 混合速度
            original_v = self.fm_v[fm_mask]
            self.fm_v[fm_mask] = m * original_v + (1 - m) * target_v
        
        # 2. 处理LASA数据
        lasa_dists = torch.norm(self.lasa_x, dim=1)
        lasa_mask = (lasa_dists > r1) & (lasa_dists < r2)
        
        if torch.any(lasa_mask):
            # 计算目标速度（指向终点）
            target_v = -speed_scale * self.lasa_x[lasa_mask]
            
            # 计算混合系数
            m = (lasa_dists[lasa_mask] - r1) / (r2 - r1)
            m = m.unsqueeze(1)  # 扩展维度用于广播
            
            # 混合速度
            original_v = self.lasa_v[lasa_mask]
            self.lasa_v[lasa_mask] = m * original_v + (1 - m) * target_v


class FMPotentialDataset(Dataset):
    def __init__(self, x, v, p):
        self.x = x.reshape(-1, 2)
        self.v = v.reshape(-1, 2)
        self.p = p.reshape(-1, 1)
        self.r, self.theta = self.cartesian_to_polar(self.x[:, 0], self.x[:, 1])
        self.normal_r = self.compute_r_normal_potential(self.p)
        self.normal_x = self.polar_to_cartesian(self.normal_r, self.theta)
        self.plottraj()

        # self.x = torch.from_numpy(x).to(device) 
        # self.v = torch.from_numpy(v).to(device)
        # self.p = torch.from_numpy(p).to(device)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.normal_x[idx]

    def cartesian_to_polar(self, x, y):
        """将笛卡尔坐标(x,y)转换为极坐标(r,theta)"""
        r = np.sqrt(x**2 + y**2).reshape(-1, 1)
        theta = np.arctan2(y, x).reshape(-1, 1)
        return r, theta
        
    def polar_to_cartesian(self, r, theta):
        """将极坐标(r, theta)转换为笛卡尔坐标(x, y)"""
        normal_x = r * np.cos(theta)
        normal_y = r * np.sin(theta)
        return np.concatenate([normal_x, normal_y], axis=1)

    def compute_r_normal_potential(self, p):
        normal_r = np.sqrt(2 * p)
        return normal_r

    def plottraj(self):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 6))
        plt.scatter(self.x[:, 0], self.x[:, 1], c=self.p[:, 0], cmap='viridis')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Trajectory Visualization')
        plt.axis('equal')

        # plt.figure(figsize=(8, 6))
        plt.scatter(self.normal_x[:, 0], self.normal_x[:, 1], c=self.p[:, 0], cmap='viridis')
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.title('Normalized Trajectory Visualization')
        # plt.axis('equal')
        plt.show()

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