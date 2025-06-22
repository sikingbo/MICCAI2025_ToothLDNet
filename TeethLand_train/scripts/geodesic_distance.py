import os
import trimesh
import numpy as np
import json
from scipy.spatial import cKDTree
from glob import glob
from sklearn.manifold import Isomap
from utils.data_utils import naive_read_pcd


def geo_distance_metrix(points):
    isomap = Isomap(n_components=2, n_neighbors=5, path_method="auto")
    data_2d = isomap.fit_transform(X=points)
    geo_distance_metrix = isomap.dist_matrix_  # 测地距离矩阵，shape=[n_sample,n_sample]
    return geo_distance_metrix


def write_geo_distance_metrix(pcd_file, out_file):
    pcd = naive_read_pcd(pcd_file)[0]
    distance_metrix = geo_distance_metrix(pcd)
    print("shape",pcd_file,"computed")
    np.savetxt(out_file,distance_metrix,fmt='%f',delimiter=',')


def write_keypoints_geo_distance_metrix(pcd_file, kp_idx, out_file):
    pcd = naive_read_pcd(pcd_file)[0]
    distance_metrix = geo_distance_metrix(pcd)
    distance_metrix = distance_metrix[kp_idx]
    print("shape",pcd_file,"computed")
    np.savetxt(out_file,distance_metrix,fmt='%f',delimiter=',')


def write_landmarks_geo_distance_metrix(off_file, json_file, out_file):
    mesh = trimesh.load(off_file)
    # vs = mesh.vertices
    cs = mesh.triangles_center
    landmarks = json.load(open(json_file))
    coords = [lm['coord'] for lm in landmarks]
    coords = np.array(coords)
    if coords.size == 0: return
    distance_matrix = np.linalg.norm(coords[:, np.newaxis, :] - cs[np.newaxis, :, :], axis=2)
    closest_idx = np.argmin(distance_matrix, axis=1)
    geo_dists = distance_matrix
    np.save(out_file, geo_dists)


if __name__ == '__main__':
    obj_path = 'E:/dataset/Teeth_box/data'
    box_txt = 'E:/dataset/Teeth_box/all.txt'
    i = 0
    with open(box_txt) as f:
        for line in f:
            filename = line.strip().split('_')[0]
            dir = line.strip().split('_')[1]  # ['lower', 'upper']
            tooth = filename + '_' + dir
            i = i + 1
            print(i, tooth)
            teeth_path = os.path.join(obj_path, tooth, 'patches_box')
            for t_idx in range(1, 20):
                off_file = os.path.join(teeth_path,  f'{t_idx}.off')
                json_file = os.path.join(teeth_path, f'{t_idx}.json')
                write_file = os.path.join(teeth_path, f'{t_idx}.npy')
                if os.path.exists(off_file) and os.path.exists(json_file):
                    write_landmarks_geo_distance_metrix(off_file, json_file, write_file)


