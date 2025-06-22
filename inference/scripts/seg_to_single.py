import os
import json
import trimesh
import numpy as np
from scripts.functions import adjust_mesh_faces

def segment_patch_box(mesh, labels):
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
    return patch_mesh


def segment_patch(mesh, labels):
    vs = mesh.vertices
    fs = mesh.faces

    fv_labels = labels[fs]
    meshs = []
    for l in range(1, 17):
        v_idx = np.argwhere(labels == l).squeeze()
        if v_idx.shape[0] == 0:
            mesh_temp = trimesh.Trimesh()
            meshs.append(mesh_temp)
            continue
        vl = vs[v_idx]

        f_idx = np.where(np.all(fv_labels == l, axis=1))[0]
        fl = fs[f_idx]

        index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(v_idx)}
        new_faces = np.array([[index_mapping[vertex] for vertex in face] for face in fl])
        patch_mesh = trimesh.Trimesh(vertices=vl, faces=new_faces)
        meshs.append(patch_mesh)
    return meshs


def tooth_landmarks(teeth_path, j_file):
    tooth_files = []
    annots = json.load(open(j_file))
    landmarks = annots['objects']
    for file in os.listdir(teeth_path):
        if file.endswith('.off'):
            file_path = os.path.join(teeth_path, file)
            tooth_files.append(file_path)
    t_idx_list = []
    for lm_dict in landmarks:
        cate = lm_dict['class']
        coord = np.array(lm_dict['coord'])
        t_idxs = []
        dists = np.ones(len(tooth_files))
        for i, f in enumerate(tooth_files):
            t_filename = os.path.basename(f)
            t_idx = os.path.splitext(t_filename)[0]
            mesh = trimesh.load(f)
            vs = mesh.vertices
            ds = np.linalg.norm(vs - coord, axis=1)
            min_dist = np.min(ds)
            dists[i] = min_dist
            t_idxs.append(t_idx)
        min_dist_idx = np.argmin(dists)
        t_idx = t_idxs[min_dist_idx]
        t_idx_list.append(int(t_idx))
    t_idx_list = np.array(t_idx_list)
    for f in tooth_files:
        t_filename = os.path.basename(f)
        t_idx = os.path.splitext(t_filename)[0]
        lm_idx = np.argwhere(t_idx_list == int(t_idx)).squeeze()
        if lm_idx.size == 0:continue
        elif lm_idx.size == 1: dict_list = [landmarks[lm_idx]]
        else: dict_list = [landmarks[idx] for idx in lm_idx]
        write_file = os.path.join(teeth_path, t_idx + '.json')
        with open(write_file, 'w') as json_file:
            json.dump(dict_list, json_file, indent=len(dict_list))


import glob
if __name__ == "__main__":
    json_path = 'D:/dataset/3DTeethLand/Batch_2_4_23_24'
    split_file = 'D:/dataset/Teeth3DS/results/test.txt'
    patch_path = 'D:/dataset/Teeth3DS/results/patch'

    with open(split_file) as f:
        for i, line in enumerate(f):
            # if i < 98:continue
            full_filename = line.strip()
            print(full_filename)
            filename = line.strip().split('_')[0]
            category = line.strip().split('_')[1]
            # root = os.path.join(mesh_path, category, filename)
            # mesh_file = os.path.join(root, f'{line.strip()}.obj')
            # seg_file = os.path.join(root, f'{line.strip()}.json')
            # land_file = os.path.join(self.args.anno_dir, f'{line.strip()}__kps.json')
            teeth_path = os.path.join(patch_path, full_filename)
            json_file = os.path.join(json_path, full_filename + '__kpt.json')
            tooth_landmarks(teeth_path, json_file)



