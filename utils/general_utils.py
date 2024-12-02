#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os.path


import math
import torch
import sys
from datetime import datetime
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors ,LocalOutlierFactor
import matplotlib.pyplot as plt
def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def PILtoTorch_d(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL))
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))


def project_to_plane(points, plane_point, plane_normal):
    """
    将点投影到平面上。
    points: (N, 3) 点云坐标
    plane_point: (3,) 平面上的一点
    plane_normal: (3,) 平面法向量
    """
    plane_normal = plane_normal / np.linalg.norm(plane_normal)  # 确保法向量是单位向量
    projected_points = points - np.dot(points - plane_point, plane_normal)[:, None] * plane_normal
    return projected_points


def knn_in_plane(points, normals, query_point, query_normal, k=10, top_n=4):
    """
    通过 KNN 找到查询点附近的 10 个点，然后在平面上找到距离查询点最近的 4 个点，
    最后比较查询点的法向量和这 4 个最近邻点的法向量之间的距离。

    points: (N, 3) 点云坐标
    normals: (N, 3) 点云法向量
    query_point: (3,) 查询点的坐标
    query_normal: (3,) 查询点的法向量
    k: 选择最近邻点的数量
    top_n: 平面上选择的最近邻点的数量
    """

    points = np.asarray(points)
    normals = np.asarray(normals)
    # 使用 KNN 找到查询点附近的 10 个点 球树方法
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(points)
    distances, indices = nbrs.kneighbors([query_point])

    # 获取最近邻的 10 个点及其法向量
    knn_points = points[indices[0]]
    knn_normals = normals[indices[0]]

    # 利用这 30 个点构建局部平面，并将它们投影到平面上
    projected_points = project_to_plane(knn_points, query_point, query_normal)
    projected_query_point = project_to_plane(query_point[np.newaxis, :], query_point, query_normal)[0]

    # 在平面上找到距离查询点最近的 5 个点
    nbrs_plane = NearestNeighbors(n_neighbors=top_n, algorithm='ball_tree').fit(projected_points)
    plane_distances, plane_indices = nbrs_plane.kneighbors([projected_query_point])

    # 获取平面上最近的 5 个点的法向量
    nearest_normals = knn_normals[plane_indices[0]]

    # # 比较查询点的法向量和这 5 个最近邻点的法向量的距离
    # normal_distances = np.linalg.norm(nearest_normals - query_normal, axis=1)

    # 返回平面上距离最近的 KNN 索引和法向量距离
    return indices[0][plane_indices[0]]


def distance_between_points(A, B):
    x1, y1, z1 = A
    x2, y2, z2 = B
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    return distance

def manhattan_distance(A, B):
    x1, y1, z1 = A
    x2, y2, z2 = B
    distance = abs(x2 - x1) + abs(y2 - y1) + abs(z2 - z1)
    return distance



def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def dynamic_knn_point_cloud(point_cloud, query_point, initial_k=3, confidence_threshold=0.7, max_k=20):
    k = initial_k
    distances = [euclidean_distance(query_point, point) for point in point_cloud]
    sorted_indices = np.argsort(distances)

    while k <= max_k:
        k_nearest_indices = sorted_indices[:k]
        k_nearest_points = point_cloud[k_nearest_indices]

        # 计算平均距离（或其他统计信息）来决定是否增加K值
        avg_distance = np.mean([distances[i] for i in k_nearest_indices])

        # 简单策略：假设置信度阈值基于距离的分布（这里使用平均距离作为简化示例）
        if avg_distance < confidence_threshold:
            return k_nearest_indices, k_nearest_points

        k += 1

    # 如果达到最大K值，返回最后的K个最近邻
    return k_nearest_indices, k_nearest_points





def detect_outliers_lof(points, n_neighbors=10, contamination=0.001, visualize=False):
    """
    使用局部异常因子（LOF）检测点云中的离群点。

    参数:
    - points: numpy.ndarray, 点云数据，形状为 (n_points, 3)
    - n_neighbors: int, 用于计算局部密度的邻居数量
    - contamination: float, 数据集中离群点的比例
    - visualize: bool, 是否可视化结果

    返回:
    - inlier_indices: numpy.ndarray, 正常点的索引数组
    - outlier_indices: numpy.ndarray, 离群点的索引数组
    """
    # 创建LOF模型
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    points = points.detach().cpu().numpy()
    # 进行训练和预测
    y_pred = lof.fit_predict(points)
    # 将预测结果转换为布尔数组
    y_pred_bool = y_pred == 1
    # 获取正常点和离群点的索引
    outlier_indices = np.where(y_pred == -1)[0]
    inlier_indices = np.where(y_pred == 1)[0]

    # 可视化结果
    # if visualize:
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.scatter(points[inlier_indices, 0], points[inlier_indices, 1], points[inlier_indices, 2], c='b', marker='o',
    #                label='Inliers')
    #     ax.scatter(points[outlier_indices, 0], points[outlier_indices, 1], points[outlier_indices, 2], c='r',
    #                marker='x', label='Outliers')
    #     ax.set_title('LOF Outlier Detection')
    #     plt.legend()
    #     plt.show()

    return y_pred_bool


def point_to_plane_distance(A, normal, B):
    # 提取点A的坐标
    x_A, y_A, z_A = A
    # 提取法向量的分量
    a, b, c = normal
    # 提取点B的坐标
    x_B, y_B, z_B = B

    # 计算平面方程的d项
    d = -(a * x_A + b * y_A + c * z_A)

    # 计算点B到平面的距离
    numerator = abs(a * x_B + b * y_B + c * z_B + d)
    denominator = np.sqrt(a ** 2 + b ** 2 + c ** 2)

    distance = numerator / denominator

    return distance