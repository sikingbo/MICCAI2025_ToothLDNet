import bpy
import os
import csv
import tqdm
import json
import click
import torch
import trimesh
import numpy as np
from mesh import TriMesh
from data.gnn.st_data import Teeth3DS
from pl_model_land import LitModel
from utils.gnn.predictor import gnn_run
from data.land.st_data import TeethLandDataset
from data.land.common import calc_features, geodesic_heatmaps, sample
from scripts.simplify import Process
from scripts.graph_cut import graph_cut
from scripts.seg_to_single import segment_patch_box
from scripts.knn import knn_map
from scripts.functions import adjust_mesh_faces


def land_run(mesh, mesh_sim, patches, name):
    checkpoint = 'runs/tooth_landmark/version_0/checkpoints/best3.ckpt'
    model = LitModel.load_from_checkpoint(checkpoint).cuda()
    model.eval()
    args = model.hparams.args

    pts_all = np.empty((0, 3))
    p_labels_all = np.array([], dtype=int)
    for patch in patches:
        if not patch.is_empty:
            vs, fs = patch.vertices, patch.faces
            vs_offset = vs.mean(0)
            vs = vs - vs_offset
            _, fids = trimesh.sample.sample_surface_even(patch, args.num_points)
            vs, fs = torch.tensor(vs, dtype=torch.float32), torch.tensor(fs[fids], dtype=torch.long)
            features1 = calc_features(vs, fs)  # (15, nf)
            features1 = features1.unsqueeze(0).cuda()
            vs_offset_t = torch.tensor(vs_offset, dtype=torch.float32).unsqueeze(0).cuda()

            vs_sim, fs_sim = mesh_sim.vertices, mesh_sim.faces
            vs_sim = torch.tensor(vs_sim, dtype=torch.float32)
            fs_sim = torch.tensor(fs_sim, dtype=torch.long)
            features2 = calc_features(vs_sim, fs_sim)  # (15, nf)
            features2 = features2.unsqueeze(0).cuda()

            with torch.no_grad():
                pts, p_labels = model.infer(features1, features2, vs_offset_t)
                pts = pts.cpu().numpy()
                pts = pts + vs_offset
                p_labels = p_labels.cpu().numpy()
                indexs = np.nonzero(p_labels)
                p_labels = p_labels[indexs]

            pts_all = np.concatenate((pts_all, pts), axis=0)
            p_labels_all = np.concatenate((p_labels_all, p_labels), axis=0)

    # pts_all = np.concatenate((pts_all, np.array([[27.2, 25.4, -91.6]])), axis=0)
    # p_labels_all = np.concatenate((p_labels_all, np.array([6])), axis=0)
    # continue
    list_csv = [None] * 6
    list_csv[0] = name
    pred_pts = [trimesh.primitives.Sphere(radius=0.7, center=pt).to_mesh() for pt in pts_all]
    for index in range(len(p_labels_all)):
        if p_labels_all[index] == 1:
            pred_pts[index].visual.vertex_colors = (255, 0, 0, 255)
            list_csv[4] = 'Mesial'
        elif p_labels_all[index] == 2:
            pred_pts[index].visual.vertex_colors = (0, 255, 0, 255)
            list_csv[4] = 'Distal'
        elif p_labels_all[index] == 3:
            pred_pts[index].visual.vertex_colors = (255, 255, 0, 255)
            list_csv[4] = 'InnerPoint'
        elif p_labels_all[index] == 4:
            pred_pts[index].visual.vertex_colors = (0, 255, 255, 255)
            list_csv[4] = 'OuterPoint'
        elif p_labels_all[index] == 5:
            pred_pts[index].visual.vertex_colors = (255, 0, 255, 255)
            list_csv[4] = 'FacialPoint'
        elif p_labels_all[index] == 6:
            pred_pts[index].visual.vertex_colors = (0, 0, 255, 255)
            list_csv[4] = 'Cusp'
        list_csv[1] = pts_all[index][0]
        list_csv[2] = pts_all[index][1]
        list_csv[3] = pts_all[index][2]
        list_csv[5] = 0.8
        with open('output/expected_output.csv', 'a', newline='', encoding='utf-8') as csvfile:
            write = csv.writer(csvfile)
            write.writerow(list_csv)

    # trimesh.Scene(pred_pts + [mesh]).show()

    COLORS = [0.7, 0.7, 0.7]
    vertex_colors = np.array([COLORS for l in range(len(mesh.vertices))])
    mesh.visual.vertex_colors = vertex_colors
    m = trimesh.Scene(pred_pts + [mesh])
    m.export('output/final' + '.obj')  # 导出为PLY格式，支持颜色


if __name__ == '__main__':
    obj_path = 'D:/dataset/Teeth_box/data'
    box_txt = 'D:/dataset/Teeth_box/temp.txt'

    header = ['key', 'coord_x', 'coord_y', 'coord_z', 'class', 'score']
    with open('output/expected_output.csv', 'w', newline='', encoding='utf-8') as csvfile:
        write = csv.writer(csvfile)
        write.writerow(header)

    with open(box_txt) as f:
        for line in f:
            filename = line.strip().split('_')[0]
            dir = line.strip().split('_')[1]  # ['lower', 'upper']
            tooth = f'{filename}_{dir}'
            mesh_file = os.path.join(obj_path, tooth, f'{tooth}.obj')
            if not os.path.exists(mesh_file):
                continue
            # mesh_path = os.path.join(obj_path, mesh_path)
            # name = os.path.basename(mesh_path)
            # name = name.strip().split('.')[0]
            mesh = trimesh.load(mesh_file)

            # simplify
            target_faces = 10000
            mesh_sim_path = os.path.join('data', 'temp', filename + '.obj')
            blender = Process(mesh_file, target_faces, mesh_sim_path)
            # print(name, "simplify OK")

            # teethgnn
            weight_dir = 'runs/all_tooth/version_0/checkpoints/best1.ckpt'
            mesh_sim = trimesh.load(mesh_sim_path)
            os.remove(mesh_sim_path)  # 删除简化后文件
            pred_sim, mesh_sim = gnn_run(mesh_sim, weight_dir)

            # knn + segment patch
            patches = []
            for i in range(len(pred_sim)):
                labels_sim = pred_sim[i]
                pred = knn_map(mesh_sim, mesh, labels_sim)
                pred = np.array(pred.flatten().tolist())
                patch = segment_patch_box(mesh, pred)
                patches.append(patch)
            # print(name, 'segment_patch ok')

            # mark land
            land_run(mesh, mesh_sim, patches, tooth)
            # print(name, 'mark land ok')


