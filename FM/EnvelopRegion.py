import numpy as np
from scipy.spatial import cKDTree
from Lasa import ImportLASADataset, filename_list
import matplotlib.pyplot as plt
import torch

class TrajectoryRegionClassifier:
    def __init__(self, trajectory_points, r1, r2):
        """
        初始化轨迹区域分类器
        
        参数:
            trajectory_points: 示教轨迹点, 形状为 [N, 2] 的 numpy 数组
            r1: 内部区域半径
            r2: 边界区域外半径 (r2 > r1)
        """
        assert r2 > r1 > 0, "半径需满足 r2 > r1 > 0"
        self.tree = cKDTree(trajectory_points)  # 构建KD树加速查询
        self.r1 = r1
        self.r2 = r2
    
    def classify_points(self, query_points):
        """
        对查询点进行分类，返回区域掩码
        
        参数:
            query_points: 待查询的点, 形状为 [M, 2] 的 numpy 数组
            
        返回:
            inner_mask: 内部区域掩码 (r < r1)
            boundary_mask: 边界区域掩码 (r1 ≤ r ≤ r2)
        """
        # 第一步：查询每个点到最近轨迹点的距离
        dists, _ = self.tree.query(query_points, k=1)
        
        # 第二步：根据距离计算区域掩码
        inner_mask = dists < self.r1
        boundary_mask = (self.r1 <= dists) & (dists <= self.r2)
        outer_mask = dists > self.r1

        return inner_mask, boundary_mask, outer_mask, dists

    def get_points_in_region(self, query_points, region_type, return_dists=False):
        """
        获取指定区域内的点及其索引
        
        参数:
            query_points: 待查询的点, 形状为 [M, 2] 的 numpy 数组
            region_type: 区域类型 ('inner' 或 'boundary')
            
        返回:
            indices: 在指定区域内的点的索引
            points: 在指定区域内的点
        """
        inner_mask, boundary_mask, outer_mask, dists = self.classify_points(query_points)

        if region_type == 'inner':
            mask = inner_mask
        elif region_type == 'boundary':
            mask = boundary_mask
        elif region_type == 'outer':
            mask = outer_mask
        else:
            raise ValueError("region_type 必须是 'inner' 或 'boundary'")
        
        indices = np.where(mask)[0]
        region_points = query_points[indices]
        
        if return_dists:
            region_dists = dists[indices]
            return indices, region_points, region_dists
        else:
            return indices, region_points
    
    def sample_outer_points(self, batch_size, limit):
        """在盆地外采样点"""
        # 在边界附近采样
        points = []
        while len(points) < batch_size:
            # 在整个工作空间采样
            x = np.random.uniform(low=[limit[0], limit[2]], high=[limit[1], limit[3]], size=(1000, 2))
            _, outer_points = self.get_points_in_region(x, 'outer')
            # 添加到列表
            points.extend(outer_points.tolist())
            # 如果足够则退出
            if len(points) >= batch_size:
                break
        # 截取所需数量的点
        points = np.array(points[:batch_size])
        return torch.from_numpy(points)

    def sample_source_points(self, batch_size, limit):
        points = np.random.uniform(low=[limit[0], limit[2]], high=[limit[1], limit[3]], size=(batch_size, 2))
        return torch.from_numpy(points)

    def sample_inner_points(self, batch_size, limit):
        """在盆地内采样点"""
        points = []
        while len(points) < batch_size:
            # 在整个工作空间采样
            x = np.random.uniform(low=[limit[0], limit[2]], high=[limit[1], limit[3]], size=(1000, 2))
            inner_mask, boundary_mask, outer_mask, dists = self.classify_points(x)
            _, inner_points = self.get_points_in_region(x, 'inner')
            # 添加到列表
            points.extend(inner_points.tolist())
            # 如果足够则退出
            if len(points) >= batch_size:
                break
        # 截取所需数量的点
        points = np.array(points[:batch_size])
        return torch.from_numpy(points)

# 使用示例
if __name__ == "__main__":

    # for file_name in filename_list:
    #     # 1. 实际数据
    #     # file_name=filename_list[0]
    #     positions, velocities, indices = ImportLASADataset(f'{file_name}.mat')
    #     x_min, x_max = positions[:, 0].min() - 20, positions[:, 0].max() + 20
    #     y_min, y_max = positions[:, 1].min() - 20, positions[:, 1].max() + 20
    #     limit=[x_min, x_max, y_min, y_max]

    #     # 2. 初始化分类器
    #     r1 = 2  # 内部区域半径
    #     r2 = 5   # 边界区域外半径
    #     classifier = TrajectoryRegionClassifier(positions, r1, r2)
        
    #     # 3. 创建测试点 (替换为您的实际查询点)
    #     test_points = np.random.uniform(low=[limit[0], limit[2]], high=[limit[1], limit[3]], size=(10000, 2))
        
    #     # 4. 进行分类查询
    #     inner_mask, boundary_mask, outer_mask = classifier.classify_points(test_points)

    #     # 5. 获取区域内的点
    #     inner_indices, inner_points = classifier.get_points_in_region(test_points, 'inner')
    #     boundary_indices, boundary_points = classifier.get_points_in_region(test_points, 'boundary')

    #     plt.figure(figsize=(10, 8))
    #     plt.scatter(positions[:, 0], positions[:, 1], c='black', s=5, label='Trajectory Points')
    #     plt.scatter(test_points[:, 0], test_points[:, 1], c='blue', s=5, label='Test Points')
    #     plt.scatter(inner_points[:, 0], inner_points[:, 1], c='red', s=3, label='Inner Points')
    #     plt.scatter(boundary_points[:, 0], boundary_points[:, 1], c='green', s=3, label='Boundary Points')
    #     plt.legend()
    #     # plt.show()
    #     plt.savefig(f'visualizations/EnvelopRegion/ER_{file_name}.png', dpi=300)


    for file_name in filename_list:
        # 1. 实际数据
        # file_name=filename_list[0]
        positions, velocities, indices = ImportLASADataset(f'{file_name}.mat')
        x_min, x_max = positions[:, 0].min() - 20, positions[:, 0].max() + 20
        y_min, y_max = positions[:, 1].min() - 20, positions[:, 1].max() + 20
        limit=[x_min, x_max, y_min, y_max]

        # 2. 初始化分类器
        r1 = 2  # 内部区域半径
        r2 = 5   # 边界区域外半径
        classifier = TrajectoryRegionClassifier(positions, r1, r2)

        # 4. 获取区域内的点
        source_points = classifier.sample_source_points(10000, limit)
        inner_points = classifier.sample_inner_points(10000, limit)

        plt.figure(figsize=(10, 8))
        plt.scatter(positions[:, 0], positions[:, 1], c='black', s=5, label='Trajectory Points')
        plt.scatter(source_points[:, 0], source_points[:, 1], c='blue', s=5, label='Source Points')
        plt.scatter(inner_points[:, 0], inner_points[:, 1], c='red', s=3, label='Inner Points')
        plt.legend()
        # plt.show()
        plt.savefig(f'visualizations/EnvelopRegion/InOut_ER_{file_name}.png', dpi=300)