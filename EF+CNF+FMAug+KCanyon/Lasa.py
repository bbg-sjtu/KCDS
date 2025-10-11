from xml.parsers.expat import model
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os


file_list1_1=[
    'AnglePy',
    'CShapePy',
    'GShapePy',
    'JShapePy',
    'KhameshPy',
    'LShapePy',
    'LinePy',
    'NShapePy',
    'PShapePy',
    'RShapePy',
    'SaeghePy',
    'SharpcPy',
    'SinePy',
    'SpoonPy',
    'SshapePy',
    'TrapezoidPy',
    'WShapePy',
    'WormPy',
    'ZshapePy',
]
# h64 l5
file_list1=[
    'AnglePy',
    'CShapePy',
    'GShapePy',
    'JShapePy',
    'KhameshPy',
    'LShapePy',
    'LinePy',
    'Multi_Models_1Py',
    'Multi_Models_2Py',
    'NShapePy',
    'PShapePy',
    'RShapePy',
    'SaeghePy',
    'SharpcPy',
    'SinePy',
    'SpoonPy',
    'SshapePy',
    'TrapezoidPy',
    'WShapePy',
    'WormPy',
    'ZshapePy',
]
# res h64 l5
file_list2=[
    'BendedLinePy',
    'Leaf_1Py',
    'Multi_Models_3Py',
    'Multi_Models_4Py',
]
# res h128 l6
file_list3=[
    'Leaf_2Py',
    'SnakePy',
]
# res h256 l6
file_list4=[
    'DoubleBendedLinePy',
    'heeePy'
]

filename_list=[
    'AnglePy',
    'BendedLinePy',
    'CShapePy',
    'DoubleBendedLinePy',
    'GShapePy',
    'JShapePy',
    'KhameshPy',
    'LShapePy',
    'Leaf_1Py',
    'Leaf_2Py',
    'LinePy',
    'Multi_Models_1Py',
    'Multi_Models_2Py',
    'Multi_Models_3Py',
    'Multi_Models_4Py',
    'NShapePy',
    'PShapePy',
    'RShapePy',
    'SaeghePy',
    'SharpcPy',
    'SinePy',
    'SnakePy',
    'SpoonPy',
    'SshapePy',
    'TrapezoidPy',
    'WShapePy',
    'WormPy',
    'ZshapePy',
    'heeePy'
]

# 函数：读取LASA数据集的数据
def ImportLASADataset(filename):
    "所有数据集文件均放在LASADataset文件夹下"
    filepath = "LASADataset/" + filename
    mat_data = scipy.io.loadmat(filepath)
    pos = np.transpose(mat_data['pos'], (2, 0, 1))
    pos = pos.reshape(-1, 2)
    vel = np.transpose(mat_data['vel'], (2, 0, 1))
    vel = vel.reshape(-1, 2)
    trajectory_ends = [999, 1999, 2999, 3999, 4999, 5999, 6999]
    vel[trajectory_ends, :] = np.zeros((len(trajectory_ends), 2), dtype=float)

    # 创建indices数组，轨迹终点为1，其他为0
    indices = np.zeros(pos.shape[0], dtype=int)  # 初始化为全0
    indices[trajectory_ends] = 1  # 将终点位置设为1

    return pos, vel, indices

    # return torch.tensor(pos, dtype=torch.float64), torch.tensor(vel, dtype=torch.float64), torch.tensor(indices, dtype=torch.long)



if __name__ == "__main__":
    filename = 'AnglePy'

    Pos_train, Vel_train, indices = ImportLASADataset(f'{filename}.mat')
    print("yes")