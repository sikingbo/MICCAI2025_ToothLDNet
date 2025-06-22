import trimesh
import json
import numpy as np
import os
from mesh import TriMesh
from scripts.graph_cut import graph_cut


def compute_oriented_bbox(keypoints):
    """
    根据关键点构建方向性包围盒（非轴对齐）。
    """
    # 提取关键点
    mesial = next((kp['coord'] for kp in keypoints if kp['class'] == 'Mesial'), None)
    distal = next((kp['coord'] for kp in keypoints if kp['class'] == 'Distal'), None)
    inner = next((kp['coord'] for kp in keypoints if kp['class'] == 'InnerPoint'), None)
    outer = next((kp['coord'] for kp in keypoints if kp['class'] == 'OuterPoint'), None)

    # 将关键点转换为 numpy 数组
    mesial = np.array(mesial) if mesial is not None else None
    distal = np.array(distal) if distal is not None else None
    inner = np.array(inner) if inner is not None else None
    outer = np.array(outer) if outer is not None else None

    all_points = np.array([kp['coord'] for kp in keypoints])
    # 中心点计算
    if mesial is not None and distal is not None and inner is not None and outer is not None:
        center = (mesial + distal + inner + outer) / 4
    else:
        center = np.mean(all_points, axis=0)

    # 初始化左右和前后方向(默认)
    left_right_dir, left_right_length = np.array([1.0, 0, 0]), 1.0
    front_back_dir, front_back_length = np.array([0, 1.0, 0]), 1.0

    # 处理左右方向
    if mesial is not None and distal is not None:
        left_right_dir = (distal - mesial) / np.linalg.norm(distal - mesial)
        left_right_length = np.linalg.norm(distal - mesial)
    elif mesial is not None:  # 如果 Distal 不存在
        if inner is not None:
            left_right_dir = np.cross(inner - mesial, [0, 0, 1])
            left_right_length = np.linalg.norm(inner - mesial)
        elif outer is not None:
            left_right_dir = np.cross(outer - mesial, [0, 0, 1])
            left_right_length = np.linalg.norm(outer - mesial)
        left_right_dir /= np.linalg.norm(left_right_dir)
    elif distal is not None:  # 如果 Mesial 不存在
        if inner is not None:
            left_right_dir = np.cross(inner - distal, [0, 0, 1])
            left_right_length = np.linalg.norm(inner - distal)
        elif outer is not None:
            left_right_dir = np.cross(outer - distal, [0, 0, 1])
            left_right_length = np.linalg.norm(outer - distal)
        left_right_dir /= np.linalg.norm(left_right_dir)

    # 处理前后方向
    if inner is not None and outer is not None:
        front_back_dir = (outer - inner) / np.linalg.norm(outer - inner)
        front_back_length = np.linalg.norm(outer - inner)
    elif inner is not None:  # 如果 Outer 不存在
        front_back_dir = np.cross([0, 0, 1], left_right_dir)
        front_back_dir /= np.linalg.norm(front_back_dir)
        front_back_length = left_right_length
    elif outer is not None:  # 如果 Inner 不存在
        front_back_dir = np.cross(left_right_dir, [0, 0, 1])
        front_back_dir /= np.linalg.norm(front_back_dir)
        front_back_length = left_right_length

    # 计算上下方向
    z_dir = np.cross(left_right_dir, front_back_dir)
    z_dir /= np.linalg.norm(z_dir)

    # 确保 Z 方向为正
    if z_dir[2] < 0:
        z_dir = -z_dir

    # 确定上下方向的范围
    z_min = np.min(all_points @ z_dir)
    z_max = np.max(all_points @ z_dir)
    z_length = z_max - z_min


    # 包围盒尺寸
    scale = 1.2  # 扩展10%
    half_sizes = scale * np.array([left_right_length / 2, front_back_length / 2, z_length / 2])

    for point in [mesial, distal, inner, outer]:
        if point is not None:
            # 投影到三个方向
            proj_x = np.dot(point - center, left_right_dir)
            proj_y = np.dot(point - center, front_back_dir)
            proj_z = np.dot(point - center, z_dir)

            # 检查并调整左右方向
            while abs(proj_x) > half_sizes[0]:
                half_sizes[0] = abs(proj_x) * 1.05  # 超出后增加 5% 边界

            # 检查并调整前后方向
            while abs(proj_y) > half_sizes[1]:
                half_sizes[1] = abs(proj_y) * 1.05  # 超出后增加 5% 边界

            # 检查并调整上下方向
            while abs(proj_z) > half_sizes[2]:
                half_sizes[2] = abs(proj_z) * 1.05  # 超出后增加 5% 边界

    # 包围盒的顶点（非轴对齐）
    box_corners = []
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 3]:
                corner = (
                        center +
                        x * half_sizes[0] * left_right_dir +
                        y * half_sizes[1] * front_back_dir +
                        z * half_sizes[2] * z_dir
                )
                box_corners.append(corner)

    return np.array(box_corners)


if __name__ == '__main__':
    # dir = ['lower', 'upper']
    obj_path = 'D:/dataset/Teeth_box/data'
    box_txt = 'D:/dataset/Teeth_box/all.txt'
    i = 1
    with open(box_txt) as f:
        for line in f:
            filename = line.strip().split('_')[0]
            dir = line.strip().split('_')[1]  # ['lower', 'upper']
            tooth = filename + '_' + dir
            obj_file = os.path.join(obj_path, tooth, tooth + '_sim.off')
            json_path = os.path.join(obj_path, tooth, 'patches')
            output_file = os.path.join(obj_path, tooth, tooth + '_sim_box.txt')
            if not os.path.exists(obj_file):
                continue

            mesh = trimesh.load(obj_file)
            points = mesh.triangles_center
            all_labels = []
            # 遍历每个 JSON 文件
            for idx in range(0, 17):
                json_file = os.path.join(json_path, str(idx) + '.json')
                if not (os.path.exists(json_file) and os.path.exists(obj_file)):
                    continue
                with open(json_file, 'r') as f:
                    keypoints = json.load(f)
                if len(keypoints) == 0:
                    continue
                oriented_bbox = compute_oriented_bbox(keypoints)  # 计算方向性包围盒
                bbox_mesh = trimesh.convex.convex_hull(oriented_bbox)  # 构建包围盒三角网格
                # 判断点是否在包围盒内
                is_inside = bbox_mesh.contains(points)
                labels = np.zeros(len(points), dtype=int)  # 初始化标签
                labels[is_inside] = 1
                # graph_cut
                # labels = graph_cut(mesh.faces, mesh.triangles_center, mesh.face_normals, labels)
                # labels = np.squeeze(labels)
                # TriMesh(mesh.vertices, mesh.faces, labels).visualize()


            all_labels = np.array(all_labels)
            # np.savetxt(output_file, all_labels, fmt='%d')  # 保存标签到 txt 文件
            print(i)
            i = i + 1
