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

    return angles, centroids_original_scale