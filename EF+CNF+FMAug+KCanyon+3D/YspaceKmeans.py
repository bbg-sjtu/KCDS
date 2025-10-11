import numpy as np
from utils import *
from Canyon_models import *
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import math


def yspace_kmeans(y_star,y0s,K):
    # 使用K-means自动聚类
    # 数据标准化（K-Means对数据尺度敏感，建议标准化）
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(y0s)

    # 创建K-Means模型并拟合数据
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    kmeans.fit(data_scaled)
    # 获取聚类结果
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    # 将中心点转换回原始数据尺度（如果需要）
    centroids_original_scale = scaler.inverse_transform(centroids)

    # 对每组聚类中心点，求取它相对于y_star的角度
    angles = []
    for centroid in centroids_original_scale:
        angle = np.arctan2(centroid[1] - y_star[0, 1], centroid[0] - y_star[0, 0])
        if angle < 0:
            angle = angle + 2 * math.pi
        angles.append(angle)
    angles = np.array(angles)

    # # 可视化结果
    # plt.figure(figsize=(10, 6))
    # # 绘制原始数据点，按聚类结果着色
    # plt.scatter(y0s[:, 0], y0s[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
    # # 绘制聚类中心
    # plt.scatter(centroids_original_scale[:, 0], centroids_original_scale[:, 1], c='red', marker='X', s=200, label='Centroids')
    # plt.title('K-Means Clustering Results')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    return angles


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import math

def yspace_kmeans_3d(y_star, y0s, K):
    """
    在三维空间中使用K-means自动聚类，并计算与KCanyon3D兼容的角度
    
    参数:
        y_star: 参考点，形状为(1, 3)
        y0s: 数据点，形状为(N, 3)
        K: 聚类数量
    
    返回:
        directions: 方向列表，每个方向为(theta, phi)元组
    """
    # 数据标准化（K-Means对数据尺度敏感，建议标准化）
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(y0s)

    # 创建K-Means模型并拟合数据
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    kmeans.fit(data_scaled)
    
    # 获取聚类结果
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    
    # 将中心点转换回原始数据尺度
    centroids_original_scale = scaler.inverse_transform(centroids)

    # 对每组聚类中心点，计算相对于y_star的方向（球坐标）
    directions = []
    for centroid in centroids_original_scale:
        # 计算从y_star指向聚类中心的向量
        vector = centroid - y_star.flatten()
        
        # 计算距离
        r = np.linalg.norm(vector)
        
        if r == 0:
            # 如果聚类中心与y_star重合，使用默认方向
            directions.append((0, 0))
            continue
            
        # 计算极角θ（与z轴的夹角）
        theta = np.arccos(vector[2] / r)
        
        # 计算方位角φ（在xy平面上的投影与x轴的夹角）
        phi = np.arctan2(vector[1], vector[0])
        
        # 确保φ在[0, 2π)范围内
        if phi < 0:
            phi += 2 * np.pi
            
        directions.append((theta, phi))

    # # 可视化结果
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # # 绘制原始数据点，按聚类结果着色
    # ax.scatter(y0s[:, 0], y0s[:, 1], y0s[:, 2], c='black')
    # # 绘制聚类中心
    # ax.scatter(centroids_original_scale[:, 0], centroids_original_scale[:, 1], centroids_original_scale[:, 2], c='red', marker='X')
    # plt.show()
    
    return directions

# 使用示例
if __name__ == "__main__":
    # 生成一些示例数据
    np.random.seed(42)
    y_star = np.array([[0, 0, 0]])  # 原点
    y0s = np.random.randn(100, 3)   # 100个随机三维点
    
    # 调用函数
    directions = yspace_kmeans_3d(y_star, y0s, 4)
    print("计算得到的方向:")
    for i, (theta, phi) in enumerate(directions):
        print(f"方向 {i+1}: θ={theta:.3f}, φ={phi:.3f}")