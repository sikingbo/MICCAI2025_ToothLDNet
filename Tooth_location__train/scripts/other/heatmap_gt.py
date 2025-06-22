import os
import trimesh
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors


def geodesic_heatmaps(geodesic_matrix, landmark_std):
    ys = []
    for i, dist in enumerate(geodesic_matrix):
        y = np.exp(- dist ** 2 / (2 * landmark_std ** 2))
        ys.append(y)
    return np.asarray(ys)


if __name__ == '__main__':
    teeth_path = 'D:/dataset/Teeth_box/data'
    box_txt = 'D:/dataset/Teeth_box/all.txt'
    patches_path = 'D:/dataset/3DTeethLand/patches'
    t = 1
    with open(box_txt) as f:
        for line in f:
            filename = line.strip().split('_')[0]
            dir = line.strip().split('_')[1]  # ['lower', 'upper']
            teeth = filename + '_' + dir
            teeth_file = os.path.join(teeth_path, teeth, teeth + '.obj')
            landmark_files = os.path.join(patches_path, teeth)
            mesh = trimesh.load(teeth_file)
            cs_mesh = mesh.triangles_center
            for k in range(1, 17):
                patch_file = os.path.join(landmark_files, str(k) + '.off')
                npy_file = os.path.join(landmark_files, str(k) + '_f.npy')
                json_file = os.path.join(landmark_files, str(k) + '.json')
                write_file = os.path.join(teeth_path, teeth, 'patches', str(k) + '_h.json')
                if not (os.path.exists(patch_file) and os.path.exists(npy_file) and os.path.exists(json_file)):
                    continue
                patch = trimesh.load(patch_file)
                landmarks = json.load(open(json_file))
                geo_dists = np.load(npy_file)
                heatmaps = geodesic_heatmaps(geo_dists, 0.5)

                cs = patch.triangles_center
                threshold = 0.000001  # 阈值
                n = heatmaps.shape[1]
                result = []  # 准备结果
                mesh_tree = cKDTree(cs_mesh)
                for idx, landmark in enumerate(landmarks):
                    heatmap = heatmaps[idx]  # 获取该关键点的热图
                    keypoint_dict = {}  # 创建一个空字典，用来存储每个面下标和热图值的映射
                    # 遍历每个热图值
                    for i in range(n):
                        if heatmap[i] >= threshold:  # 如果热图值大于等于阈值
                            point_coords = cs[i]  # 找到该点的坐标
                            dist, nearest_idx = mesh_tree.query(point_coords)  # 在 mesh_tree 中查找距离该点最近的网格点的下标
                            keypoint_dict[nearest_idx] = heatmap[i]  # 将该网格点下标和热图值存储到字典中
                    # 将该关键点的字典添加到结果列表中
                    result.append({
                        "class": landmark["class"],  # 保留关键点的名称
                        "heatmap_values": keypoint_dict  # 存储该关键点的字典
                    })
                with open(write_file, 'w') as f:
                    json.dump(result, f, indent=4)
            print(t)
            t = t + 1

