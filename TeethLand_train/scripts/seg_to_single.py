import os
import json
import trimesh
import numpy as np

tar_labels_upper = [11,12,13,14,15,16,17,18,21,22,23,24,25,26,27,28]
tar_labels_lower = [31,32,33,34,35,36,37,38,41,42,43,44,45,46,47,48]


def trans_labels(labels, category):
    if category == 'upper':
        for i, l in enumerate(tar_labels_upper):
            idx = np.argwhere(labels == l)
            labels[idx] = i + 1
    else:
        for i, l in enumerate(tar_labels_lower):
            idx = np.argwhere(labels == l)
            labels[idx] = i + 1
    return labels


def save_split(json_path, save_file):

    with open(save_file, 'w') as f:
        for file in os.listdir(json_path):
            if file.endswith('.json'):
                filename = file.strip().split('_')[0]
                category = file.strip().split('_')[1]
                filename_all = filename + '_' + category
                f.write(filename_all + '\n')


def segment_patch(mesh, labels, write_file):
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
    patch_mesh.export(write_file)


def assign_landmarks(teeth_path, j_file):
    tooth_files = []
    annots = json.load(open(j_file))
    landmarks = annots['objects']
    count_pre = len(landmarks)
    count_post = 0
    for file in os.listdir(teeth_path):
        if file.endswith('.off'):
            file_path = os.path.join(teeth_path, file)
            tooth_files.append(file_path)
    mes_coords = []
    dis_coords = []
    inner_coords = []
    outer_coords = []
    fa_coords = []
    cusp_coords = []
    for dic in landmarks:
        if dic['class'] == 'Mesial':
            mes_coords.append(dic['coord'])
        elif dic['class'] == 'Distal':
            dis_coords.append(dic['coord'])
        elif dic['class'] == 'InnerPoint':
            inner_coords.append(dic['coord'])
        elif dic['class'] == 'OuterPoint':
            outer_coords.append(dic['coord'])
        elif dic['class'] == 'FacialPoint':
            fa_coords.append(dic['coord'])
        else:
            cusp_coords.append(dic['coord'])
    mes_coords = np.array(mes_coords)
    dis_coords = np.array(dis_coords)
    inner_coords = np.array(inner_coords)
    outer_coords = np.array(outer_coords)
    fa_coords = np.array(fa_coords)
    cusp_coords = np.array(cusp_coords)
    for f in tooth_files:
        t_idx = os.path.basename(f)
        write_file = os.path.join(teeth_path, t_idx + '.json')
        t_land = []
        mesh = trimesh.load(f)
        vs = mesh.vertices

        dists = np.linalg.norm(mes_coords[:, np.newaxis, :] - vs[np.newaxis, :, :], axis=2)
        min_dists = np.min(dists, axis=1)
        min_idx = np.argmin(min_dists)
        min_mes_dist = min_dists[min_idx]
        min_mes_coord = mes_coords[min_idx]

        dists = np.linalg.norm(dis_coords[:, np.newaxis, :] - vs[np.newaxis, :, :], axis=2)
        min_dists = np.min(dists, axis=1)
        min_idx = np.argmin(min_dists)
        min_dis_dist = min_dists[min_idx]
        min_dis_coord = dis_coords[min_idx]

        dists = np.linalg.norm(inner_coords[:, np.newaxis, :] - vs[np.newaxis, :, :], axis=2)
        min_dists = np.min(dists, axis=1)
        min_idx = np.argmin(min_dists)
        min_inner_dist = min_dists[min_idx]
        min_inner_coord = inner_coords[min_idx]

        dists = np.linalg.norm(outer_coords[:, np.newaxis, :] - vs[np.newaxis, :, :], axis=2)
        min_dists = np.min(dists, axis=1)
        min_idx = np.argmin(min_dists)
        min_outer_dist = min_dists[min_idx]
        min_outer_coord = outer_coords[min_idx]

        dists = np.linalg.norm(fa_coords[:, np.newaxis, :] - vs[np.newaxis, :, :], axis=2)
        min_dists = np.min(dists, axis=1)
        min_idx = np.argmin(min_dists)
        min_fa_dist = min_dists[min_idx]
        min_fa_coord = fa_coords[min_idx]

        if min_mes_dist < 0.7:
            t_land.append({'class':'Mesial','coord':min_mes_coord.tolist()})
            count_post += 1
        if min_dis_dist < 0.7:
            t_land.append({'class':'Distal','coord':min_dis_coord.tolist()})
            count_post += 1
        if min_inner_dist < 0.7:
            t_land.append({'class':'InnerPoint','coord':min_inner_coord.tolist()})
            count_post += 1
        if min_outer_dist < 0.7:
            t_land.append({'class':'OuterPoint','coord':min_outer_coord.tolist()})
            count_post += 1
        if min_fa_dist < 0.7:
            t_land.append({'class':'FacialPoint','coord':min_fa_coord.tolist()})
            count_post += 1

        dists = np.linalg.norm(cusp_coords[:, np.newaxis, :] - vs[np.newaxis, :, :], axis=2)
        min_dists = np.min(dists, axis=1)
        cusp_idx = np.argwhere(min_dists < 0.5)
        for idx in cusp_idx:
            coord = cusp_coords[idx]
            t_land.append({'class': 'Cusp', 'coord': coord.tolist()})
            count_post += 1

        # t_land_dict = {str(i): item for i, item in enumerate(t_land)}
        with open(write_file, 'w') as json_file:
            json.dump(t_land, json_file, indent=len(t_land))
    if count_post == count_pre:
        print('True')
    elif count_post < count_pre:
        print('In')
    else:
        print('Out')


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
        dists = []
        # dists = np.ones(len(tooth_files))
        for i, f in enumerate(tooth_files):
            t_filename = os.path.basename(f)
            t_idx = os.path.splitext(t_filename)[0]
            mesh = trimesh.load(f)
            vs = mesh.vertices
            ds = np.linalg.norm(vs - coord, axis=1)
            min_dist = np.min(ds)
            dists.append(min_dist)
            t_idxs.append(t_idx)
        min_dist_idx = np.argmin(np.array(dists))
        t_idx = t_idxs[min_dist_idx]
        t_idx_list.append(int(t_idx))
    t_idx_list = np.array(t_idx_list)
    for f in tooth_files:
        t_filename = os.path.basename(f)
        t_idx = os.path.splitext(t_filename)[0]
        lm_idx = np.argwhere(t_idx_list == int(t_idx)).squeeze()
        if lm_idx.size == 0: continue
        elif lm_idx.size == 1: dict_list = [landmarks[lm_idx]]
        else: dict_list = [landmarks[idx] for idx in lm_idx]
        write_file = os.path.join(teeth_path, t_idx + '.json')
        with open(write_file, 'w') as json_file:
            json.dump(dict_list, json_file, indent=len(dict_list))

import glob
if __name__ == "__main__":
    json_path = 'E:/dataset/3DTeethLand/Batch'
    obj_path = 'E:/dataset/Teeth_box/data'
    box_txt = 'E:/dataset/Teeth_box/all.txt'
    i = 0
    with open(box_txt) as f:
        for line in f:
            i = i + 1
            filename = line.strip().split('_')[0]
            dir = line.strip().split('_')[1]  # ['lower', 'upper']
            tooth = filename + '_' + dir
            print(i, tooth)
            teeth_path = os.path.join(obj_path, tooth, 'patches_box')
            json_file = os.path.join(json_path, tooth + '__kpt.json')
            tooth_landmarks(teeth_path, json_file)

# if __name__ == "__main__":
#     mesh_path = 'F:/dataset/Teeth3DS/data'


