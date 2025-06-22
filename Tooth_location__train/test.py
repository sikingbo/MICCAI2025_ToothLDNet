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
from data.st_data import Teeth3DS
from utils.TeethGNN.cluster import Cluster
from utils.TeethGNN.metrics import get_iou, get_contour_points, get_tooth_iou
from data.common import calc_features
from scripts.graph_cut import graph_cut


@click.command()
# @click.option('--checkpoint', type=str, default='E:/code/tooth_seg/teethgnn/runs/all_tooth/version_7_lower/checkpoints/last.ckpt')
@click.option('--weight_dir', type=str, default='E:/code/teeth/box/TeethGNN_train1/runs/all_tooth/version_0/', help='')
@click.option('--gpus', default=1)
def run(weight_dir, gpus):
    write_path = 'E:/code/teeth/box/TeethGNN_train1/runs/all_tooth/version_2/checkpoints/best.ckpt'

    model = LitModel.load_from_checkpoint(write_path).cuda()
    model.eval()
    args = model.hparams.args
    dataset = Teeth3DS(args, args.test_file, False)

    for i in tqdm.tqdm(range(len(dataset))):
        # feats, _ = dataset[i]
        mesh_file, _ = dataset.files[i]
        print(mesh_file)
        mesh = trimesh.load_mesh(mesh_file)
        vs, fs = mesh.vertices, mesh.faces
        cs = mesh.triangles_center
        fn = mesh.face_normals
        # _, fids = trimesh.sample.sample_surface_even(mesh, args.num_points)
        fids = np.arange(args.num_points)
        fs, cs, fn = fs[fids], cs[fids], fn[fids]
        vs_t = torch.tensor(vs, dtype=torch.float32)
        fs_t = torch.tensor(fs, dtype=torch.long)
        features = calc_features(vs_t, fs_t)  # (15, nf)
        features = features.unsqueeze(0).cuda()

        # pred
        with torch.no_grad():
            p_labels = model.infer(features)  # M,N

        # filename = os.path.basename(mesh_file).split('.')[0]
        # write_file = os.path.join(write_path, filename + '.txt')
        # np.savetxt(write_file, p_labels, fmt='%d', delimiter=',')
        # print(write_file)

        # p_labels = np.sum(p_labels, axis=0)
        # p_labels[p_labels != 0] = 1
        # labels = graph_cut(fs, cs, fn, p_labels)
        # labels = np.squeeze(labels)
        # TriMesh(vs, fs, labels).visualize()
        s = 0
        for i in range(len(p_labels)):
            labels = p_labels[i]
            if sum(labels) == 0:
                continue
            labels = graph_cut(fs, cs, fn, labels)
            labels = np.squeeze(labels)
            if sum(labels) >= 20:
                TriMesh(vs, fs, labels).visualize()
                s = s + 1
        print(s)

if __name__ == "__main__":
    run()
