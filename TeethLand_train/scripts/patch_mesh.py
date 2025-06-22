import os
import tqdm
import click
import torch
import trimesh
import numpy as np
import torch.nn.functional as F
from mesh import TriMesh
from sklearn import metrics

from pl_model import LitModel
from data.common import calc_features
from scripts.graph_cut import graph_cut
from scripts.functions import adjust_mesh_faces, find_closest_faces
from sklearn.neighbors import NearestNeighbors


def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                clear_folder(file_path)
                os.rmdir(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


def knn_map(sim_mesh, ori_mesh, sim_pred):
    sam_c = sim_mesh.triangles_center
    ori_c = ori_mesh.vertices

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(sam_c)
    _, indices = nbrs.kneighbors(ori_c)
    indices = indices.astype(int).flatten()

    ori_pred = sim_pred[indices]
    return ori_pred


def segment_patch(mesh, labels, patches_box_path, i):
    vs = mesh.vertices
    fs = mesh.faces
    fv_labels = labels[fs]

    v_idx = np.argwhere(labels == 1).squeeze()
    vl = vs[v_idx]

    f_idx = np.where(np.all(fv_labels == 1, axis=1))[0]
    fl = fs[f_idx]

    index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(v_idx)}
    new_faces = np.array([[index_mapping[vertex] for vertex in face] for face in fl])
    patch_mesh = trimesh.Trimesh(vertices=vl, faces=new_faces)
    patch_mesh = adjust_mesh_faces(patch_mesh, target_face_count=10000)
    write_patch_file = os.path.join(patches_box_path, f'{i + 1}.off')
    patch_mesh.export(write_patch_file)

    write_map_file = os.path.join(patches_box_path, f'{i + 1}_map.npy')
    find_closest_faces(patch_mesh, mesh, write_map_file)


def run():
    obj_path = 'E:/dataset/Teeth_box/data'
    box_txt = 'E:/dataset/Teeth_box/all.txt'
    t = 0
    with open(box_txt) as f:
        for line in f:
            filename = line.strip().split('_')[0]
            dir = line.strip().split('_')[1]  # ['lower', 'upper']
            tooth = filename + '_' + dir
            root = os.path.join(obj_path, tooth)
            patches_box = os.path.join(root, 'patches_box')
            if not os.path.exists(patches_box):
                os.mkdir(patches_box)
            else:
                clear_folder(patches_box)
            mesh_file = os.path.join(root, f'{tooth}.obj')
            mesh_sim_file = os.path.join(root, f'{tooth}_sim.off')
            txt_file = os.path.join(root, f'{tooth}_sim_box_gc.txt')
            t = t + 1
            print(t, mesh_file)
            mesh = trimesh.load_mesh(mesh_file)
            mesh_sim = trimesh.load_mesh(mesh_sim_file)
            labels = np.loadtxt(txt_file, dtype=np.int32)

            for i in range(len(labels)):
                labels_sim = labels[i]
                if sum(labels_sim) == 0: continue
                labels_sim = graph_cut(mesh_sim.faces, mesh_sim.triangles_center, mesh_sim.face_normals, labels_sim)
                labels_sim = np.squeeze(labels_sim)
                if sum(labels_sim) <= 120: continue
                # TriMesh(mesh_sim.vertices, mesh_sim.faces, labels_sim).visualize()
                # knn
                pred_labels = knn_map(mesh_sim, mesh, labels_sim)
                # segment patch
                pred_labels = np.array(pred_labels.flatten().tolist())
                segment_patch(mesh, pred_labels, patches_box, i)


if __name__ == "__main__":
    run()
