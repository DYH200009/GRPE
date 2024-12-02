import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# 加载点云
point_cloud = o3d.io.read_point_cloud("F:\MODELS\SuGaR\output_1/refined_ply\grape\sugarfine_3Dgs7000_densityestim02_sdfnorm02_level03_decim1000000_normalconsistency01_gaussperface1.ply")

# 计算法向量
point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
    radius=0.1, max_nn=30))

# 获取法向量
normals = np.asarray(point_cloud.normals)

# 将法向量转换为 HSV
normals_normalized = (normals + 1) / 2  # 将值调整到 [0, 1] 区间
hsv_image = plt.cm.hsv(normals_normalized)[:, :, :3]  # 仅取 RGB 值

# 渲染 HSV 图像
plt.imshow(hsv_image)
plt.axis('off')
plt.show()
