import os
import tqdm
import click
import torch
import trimesh
import numpy as np
import torch.nn.functional as F
import time

from scipy.spatial.distance import cdist
from pl_model import LitModel
from utils.data_utils import normalize_pc, vis_result
from data.st_data import TeethLandDataset
from utils.metrics import eval_iou, get_cd, saliency_iou, hungary_iou,chamfer_distance
from visual.vis_noise import get_saliency

@click.command()
@click.option('--checkpoint', type=str, default='E:/code/teeth/box/TeethLand_DETR/runs/tooth_landmark/version_1/checkpoints/best.ckpt')
@click.option('--gpus', default=1)
def run(checkpoint, gpus):
    model = LitModel.load_from_checkpoint(checkpoint).cuda()
    model.eval()

    args = model.hparams.args
    test_file = 'E:/dataset/Teeth_box/temp.txt'
    dataset = TeethLandDataset(args, test_file, False)

    for i in tqdm.tqdm(range(len(dataset))):
        feats, _, _, _ = dataset[i]
        off_file, _, _, _ = dataset.files[i]
        # mesh
        print(off_file)
        mesh = trimesh.load(off_file)
        vs = mesh.vertices
        vs_offset = vs.mean(0)

        # sample_xyz = np.array(feats[:3].T)  # 10000 x 3
        feats = feats.unsqueeze(0).cuda()

        # pred
        with torch.no_grad():
            pts, p_labels = model.infer(feats)
            pts = pts + vs_offset

        # write_file = 'F:/dataset/keypointnet/results/chair/saliency_xyz/' + mesh_name + '.txt'
        # vis_result(pc_np, gts, pts, write_file)
        # np.savetxt(write_file, pts, fmt='%f', delimiter=',')

        # continue
        # gt_pts = [trimesh.primitives.Sphere(radius=0.2, center=pt).to_mesh() for pt in gts]
        # for pt in gt_pts:
        #     pt.visual.vertex_colors = (255, 0, 0, 255)
        pred_pts = [trimesh.primitives.Sphere(radius=0.2, center=pt).to_mesh() for pt in pts]
        for pt in pred_pts:
            pt.visual.vertex_colors = (0, 255, 0, 255)
        trimesh.Scene([mesh] + pred_pts).show()


if __name__ == "__main__":
    run()
