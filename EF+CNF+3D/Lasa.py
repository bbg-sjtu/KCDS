from xml.parsers.expat import model
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os

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


file_list_3D=[
    "3D_Cshape_bottom",
    "3D_Cshape_top",
    "3D_sink",
    "3D_sink_hty",
    "3D_sink_hty1",
    "3D_viapoint_1",
    "3D_viapoint_2",
    "3D_viapoint_3",
    "3D-cube-pick",
    "3D-pick-box"
]

file_list_3D_good=[
    "3D_Cshape_bottom",
    "3D_Cshape_top",
    "3D_sink",
    # "3D_sink_hty",
    # "3D_sink_hty1",
    "3D_viapoint_1",
    "3D_viapoint_2",
    # "3D_viapoint_3",
    "3D-cube-pick",
    # "3D-pick-box"
]

def Import3DDataset(file_name):
    "所有数据集文件均放在3DDataset文件夹下"
    filepath = "3DDataset/3D dataset modified/" + file_name
    mat_data = scipy.io.loadmat(filepath)
    len1 = mat_data['data'].shape[0]
    len2 = 0
    for i in range(len1):
        len2 += mat_data['data'][i][0].shape[1]

    data=np.zeros((len2, 6), dtype=float)
    last_idx=0
    trajectory_starts = []
    trajectory_ends = []
    for i in range(len1):
        len = mat_data['data'][i][0].shape[1]
        data[last_idx:last_idx+len, :] = np.transpose(mat_data['data'][i][0], (1, 0))
        trajectory_starts.append(last_idx)
        last_idx += len
        trajectory_ends.append(last_idx - 1)

    pos = data[:, :3]
    vel = data[:, 3:]
    # indices = np.zeros(pos.shape[0], dtype=int)
    # indices[trajectory_ends] = 1

    return pos, vel, trajectory_starts, trajectory_ends

if __name__ == "__main__":
    file_name='3D_Cshape_bottom'
    pos, vel, trajectory_starts,trajectory_ends=Import3DDataset(file_name)