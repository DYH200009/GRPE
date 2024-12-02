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

import os
import sys
from PIL import Image
import matplotlib.pyplot as plt
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation,knn_in_plane,distance_between_points,detect_outliers_lof,manhattan_distance,point_to_plane_distance
import torch
from sklearn.neighbors import NearestNeighbors ,LocalOutlierFactor
from sklearn.cluster import DBSCAN


import open3d as o3d
import random

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_depth: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, scene_scale):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        R_w_c = R
        R_c_w = np.linalg.inv(R)
        T_w_c = - R_w_c @ T * scene_scale
        T_c_w = - R_c_w @ T_w_c

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        depth_path = os.path.join(images_folder, 'depths', os.path.basename(extr.name
                                                                            .replace('frame', 'depth')
                                                                            .replace('jpg', 'png')))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        # image_depth = Image.fromarray((np.array(Image.open(depth_path))/6553.5).astype(np.float32), mode='F')
        # plt.imshow(np.array(image_depth), cmap='magma')
        cam_info = CameraInfo(uid=uid, R=R, T=T_c_w, FovY=FovY, FovX=FovX, image=image, image_depth=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    types = np.zeros((colors.shape[0], 1))

    return BasicPointCloud(points=positions, colors=colors, normals=normals, types=types)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, scene_scale, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), scene_scale=scene_scale)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:

        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        # 计算需要保留的图片数量，按照scales的比例

        # # 图像总数
        # num_train_images = len(train_cam_infos)
        #
        # # 每5张图像中保留的数量
        # num_keep_per_group = int(10 * scales)
        #
        # # 创建布尔数组来标记哪些图片需要保留
        # keep_image = []
        # for i in range(num_train_images):
        #     # 计算当前图像所在的组
        #     group_index = i // 10
        #     # 确定当前组中的位置
        #     position_in_group = i % 10
        #     # 每组中选择4个保留
        #     keep_image.append(position_in_group < num_keep_per_group)
        #
        # # 转换为布尔数组
        # keep_image = np.array(keep_image)
        #
        #
        # # 根据布尔数组保留图片
        # train_cam_infos = [train_cam_infos[i] for i in range(num_train_images) if keep_image[i]]
        # print("The number of keep image", len(train_cam_infos))
        # test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
        scales = 8 / 10
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        # 计算需要保留的图片数量，按照scales的比例
        num_train_images = len(train_cam_infos)
        num_keep_images = int(num_train_images * scales)  # 4/5的比例

        # 创建布尔数组来标记哪些图片需要保留
        keep_image = [True] * num_keep_images + [False] * (num_train_images - num_keep_images)

        # 根据布尔数组保留图片
        train_cam_infos = [train_cam_infos[i] for i in range(num_train_images) if keep_image[i]]
        print("The number of keep image", len(train_cam_infos))
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None


    plydata = PlyData.read(ply_path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

    # colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).astype(np.float32).T / 255.0

    color_map = {
        0: np.array([0, 0, 255]) / 255.0,  # 深蓝色（归一化到[0, 1]范围）
        1: np.array([255, 102, 102]) / 255.0,  # 亮红色（归一化到[0, 1]范围）
    }

    # 为每个点分配颜色
    colors = np.zeros((positions.shape[0], 3))  # 初始化颜色数组

    pcd = o3d.geometry.PointCloud()


    pcd.points = o3d.utility.Vector3dVector(positions.reshape(-1, 3))
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=10))

    normals = np.array(pcd.normals)

    types = np.zeros((colors.shape[0], 1))

    # 转换点云数据为Numpy数组
    points_np = np.asarray(pcd.points)

    # 使用DBSCAN进行聚类
    # eps 是点的最大距离，min_samples 是形成簇的最小点数
    # dbscan = DBSCAN(eps=0.05, min_samples=10)
    # labels = dbscan.fit_predict(points_np)
    # # 获取每个簇的大小
    # unique_labels, counts = np.unique(labels, return_counts=True)
    #
    # # 找到最大簇的标签（排除噪声点）
    # largest_cluster_label = unique_labels[counts.argmax()]
    #
    # # 创建掩码以筛选最大簇
    # largest_cluster_mask = labels == largest_cluster_label
    #
    # # 筛选出最大簇的点
    # filtered_points = points_np[largest_cluster_mask]
    #
    # # 创建新的点云对象，并将最大簇的点设置到其中
    # filtered_pcd = o3d.geometry.PointCloud()
    # filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    #
    # pcd = filtered_pcd
    knn_tree = o3d.geometry.KDTreeFlann(pcd)
    # 寻找最近邻点
    k = 4
    knn_indexs = [knn_tree.search_knn_vector_3d(p, knn=k)[1] for p in pcd.points]
    distance_threshold = 0.03
    for knn_index in knn_indexs:
        distance = 0
        is_valid = True
        current_normal = normals[knn_index[0]]
        current_point = pcd.points[knn_index[0]]
        for i in range(k):
            distance = distance + distance_between_points(current_point, pcd.points[knn_index[i]])
        for idx in range(k):
            if np.sum(current_normal * normals[knn_index[idx]]) < 1-np.cos(0.03):
                is_valid = False
                break
        if is_valid:
            # if distance < distance_threshold:
                for idx in range(k):
                    types[knn_index[idx]] = 1
                    # colors[knn_index[idx]] = np.array([0, 1, 0])
    # 定义颜色映射

    for i in range(len(types)):
        point_type = int(types[i])
        if point_type in color_map:
            colors[i] = color_map[point_type]  # 根据类型设置颜色
        else:
            colors[i] = np.array([0.5, 0.5, 0.5])  # 默认颜色（灰色）
            colors[i] = np.array([0.5, 0.5, 0.5])  # 默认颜色（灰色）

    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.points = o3d.utility.Vector3dVector(positions)
    o3d.io.write_point_cloud("output_DBSCAN.ply", pcd)
    o3d.visualization.draw_geometries([pcd])

    print("pc normal init over!")
    pcd = BasicPointCloud(points=positions, colors=colors, normals=normals, types=types)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,image_depth = image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))

    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    plydata = PlyData.read(ply_path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).astype(np.float32).T / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions.reshape(-1, 3))
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=10))
    # o3d.visualization.draw_geometries([pcd], point_show_normal=True)

    normals = np.array(pcd.normals)

    types = np.zeros((colors.shape[0], 1))


    print("pc normal init over!")

    pcd = BasicPointCloud(points=positions, colors=colors, normals=normals, types=types)
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCamerasFromTraj(path, trajfile, white_background, is_train=True, per_frame=5):
    cam_infos = []

    with open(os.path.join(path, trajfile)) as txt_file:
        lines = txt_file.readlines()

    count = -1
    for idx, line in enumerate(lines):
        if idx % 5 != 0:
            continue

        line = line.split()

        count += 1
        if is_train:
            if count % per_frame == 0:
                continue
        else:
            if count % per_frame != 0:
                continue


        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}".format(idx+1))
        sys.stdout.flush()


        c2w = np.array(list(map(float, line))).reshape(4, 4)
        R_w_c = c2w[:3, :3]
        T_c_w = - np.transpose(R_w_c) @ c2w[:3, 3]

        rgb_name = os.path.join(path, 'results', f'frame{idx:06d}.jpg')
        d_name = os.path.join(path, 'results', f'depth{idx:06d}.png')
        image_name = Path(rgb_name).stem
        rgb = Image.open(rgb_name)
        d =Image.fromarray((np.array(Image.open(d_name)) / 6553.5).astype(np.float32), mode='F')

        # "w": 1200,
        # "h": 680,
        # "fx": 600.0,
        # "y": 600.0,
        # "cx": 599.5,
        # "cy": 339.5,
        focal_length_x = 600.0
        focal_length_y = 600.0
        FovX = focal2fov(focal_length_x, rgb.size[0])
        FovY = focal2fov(focal_length_y, rgb.size[1])

        cam_infos.append(CameraInfo(uid=idx, R=R_w_c, T=T_c_w, FovY=FovY, FovX=FovX, image=rgb, image_depth=d,
                                    image_path=rgb_name, image_name=image_name, width=rgb.size[0],
                                    height=rgb.size[1]))

    sys.stdout.write('\n')
    return cam_infos


def readReplicaSceneInfo(path, white_background, eval):
    print("Reading Training Data")
    train_cam_infos = readCamerasFromTraj(path, "traj.txt", white_background, is_train=True)
    test_cam_infos = readCamerasFromTraj(path, "traj.txt", white_background, is_train=False)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3D.ply")
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCamerasFromKeyFrameTraj(path, trajfile, white_background, is_train=True, per_frame=5, sparse_num=1):
    cam_infos = []

    with open(os.path.join(path, trajfile)) as txt_file:
        lines = txt_file.readlines()

    count = -1
    for line in lines:
        line = line.split()

        idx = int(float(line[0]))

        count += 1

        if is_train:
            if count % per_frame == 0:
                continue
        else:
            if count % per_frame != 0:
                continue


        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}".format(idx+1))
        sys.stdout.flush()


        c2w = np.array(list(map(float, line[1:]))).reshape(4, 4)
        R_w_c = c2w[:3, :3]
        T_c_w = - np.transpose(R_w_c) @ c2w[:3, 3]

        # rgb_name = os.path.join(path, 'results', f'frame{idx:06d}.jpg')
        rgb_name = os.path.join(path, 'results', f'{idx}.png')                # icl
        # rgb_name = os.path.join(path, 'results', f'{float(line[0]):.6f}.png')    # tum
        # d_name = os.path.join(path, 'results', f'depth{idx:06d}.png')
        image_name = Path(rgb_name).stem
        rgb = Image.open(rgb_name)
        # d =Image.fromarray((np.array(Image.open(d_name)) / 6553.5).astype(np.float32), mode='F')
        d =rgb

        # "w": 1200,
        # "h": 680,
        # "fx": 600.0,
        # "y": 600.0,
        # "cx": 599.5,
        # "cy": 339.5,
        focal_length_x = 481.2 #535.4 #600.0#
        focal_length_y = 480 #539.2 #600.0#
        FovX = focal2fov(focal_length_x, rgb.size[0])
        FovY = focal2fov(focal_length_y, rgb.size[1])

        cam_infos.append(CameraInfo(uid=idx, R=R_w_c, T=T_c_w, FovY=FovY, FovX=FovX, image=rgb, image_depth=d,
                                    image_path=rgb_name, image_name=image_name, width=rgb.size[0],
                                    height=rgb.size[1]))

    sys.stdout.write('\n')

    if is_train:
        cam_infos = cam_infos[::sparse_num]
    return cam_infos



def readManhattanSceneInfo(path, white_background, eval, sparse_num):
    print("Reading Training Data")
    train_cam_infos = readCamerasFromKeyFrameTraj(path, "KeyFrameTrajectory2.txt", white_background, is_train=True, sparse_num=sparse_num)
    test_cam_infos = readCamerasFromKeyFrameTraj(path, "KeyFrameTrajectory2.txt", white_background, is_train=False)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)


    ply_path = os.path.join(path, "PointCloud.ply")

    plydata = PlyData.read(ply_path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

    normals = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).astype(np.float32).T
    mask_zero = (normals.sum(axis=1, keepdims=True) == 0)
    # mask_zero3 = mask_zero.repeat(3,axis=1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions.reshape(-1, 3))
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=10))
    # o3d.visualization.draw_geometries([pcd], point_show_normal=True)

    normals = np.array(pcd.normals)
    # normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    colors = np.zeros_like(positions)
    colors += np.array([1, 0, 0])
    # types = np.ones((colors.shape[0], 1))
    # types[mask_zero] = 0
    types = np.zeros((colors.shape[0], 1))


    knn_tree = o3d.geometry.KDTreeFlann(pcd)
    # 寻找最近邻点
    k = 5
    knn_indexs = [knn_tree.search_knn_vector_3d(p, knn=k)[1] for p in pcd.points]
    for knn_index in knn_indexs:
        is_valid = True
        current_normal = normals[knn_index[0]]
        for idx in range(k):
            if np.sum(current_normal * normals[knn_index[idx]]) < 1-np.cos(0.03):
                is_valid = False
                break
        if is_valid:
            for idx in range(k):
                types[knn_index[idx]] = 1
                # colors[knn_index[idx]] = np.array([0, 1, 0])
    # pcd = o3d.geometry.PointCloud()
    # pcd.colors = o3d.utility.Vector3dVector(colors)
    # pcd.points = o3d.utility.Vector3dVector(positions)
    # o3d.visualization.draw_geometries([pcd])

    print("pc normal init over!")
    pcd = BasicPointCloud(points=positions, colors=colors, normals=normals, types=types)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Replica" : readReplicaSceneInfo,
    "Manhattan" : readManhattanSceneInfo
}