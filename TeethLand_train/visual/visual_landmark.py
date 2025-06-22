import os
import tqdm
import json
import click
import torch
import trimesh
import numpy as np
import torch.nn.functional as F
import time

from scipy.spatial.distance import cdist
from pl_model import LitModel
from utils.data_utils import normalize_pc, vis_result
# from data.st_data import KPS_Geodesic_Dataset, NAMES2ID
from utils.metrics import eval_iou, get_cd, saliency_iou, hungary_iou,chamfer_distance
from visual.vis_noise import get_saliency

@click.command()
@click.option('--checkpoint', type=str, default='E:/code/keypoint/heat_match_classify/runs/keypoint_saliency/dy_k20/version_22_chair/checkpoints/0.9367.ckpt')
@click.option('--gpus', default=1)
def run(checkpoint, gpus):
    model = LitModel.load_from_checkpoint(checkpoint).cuda()
    model.eval()

    args = model.hparams.args
    test_file = 'F:/dataset/keypointnet/splits/test.txt'
    mesh_root = 'F:/dataset/keypointnet/ShapeNetCore.v2.ply'

    # miou = []
    # mpck = []
    mcd = []
    smiou = {}
    hmiou = {}

    # 使用循环创建字典，并为每个键分配一个空列表作为值
    for i in range(11):
        key = i * 0.01
        smiou[key] = []
        hmiou[key] = []

    for i in tqdm.tqdm(range(len(dataset))):
        pc, heat, mesh_name = dataset[i]
        # mesh
        print(mesh_name)
        mesh = trimesh.load(os.path.join(mesh_root, class_id, mesh_name + '.ply'))
        # vs = np.array(mesh.vertices, dtype='float32')
        # vs = normalize_pc(vs)
        # mesh.vertices = vs
        # pc = normalize_pc(pc)
        # pc_np = pc.T
        # pc = add_noise(pc, 0.020, 1)
        pc = torch.tensor(pc, dtype=torch.float32).unsqueeze(0).cuda()
        heat = torch.tensor(heat, dtype=torch.float32).unsqueeze(0).cuda()
        # pred
        with torch.no_grad():
            pts, gts = model.infer(pc, heat)
            pts = pts.cpu().numpy()
            gts = gts.cpu().numpy()

        # write_file = 'F:/dataset/keypointnet/results/chair/saliency_xyz/' + mesh_name + '.txt'
        # vis_result(pc_np, gts, pts, write_file)
        # np.savetxt(write_file, pts, fmt='%f', delimiter=',')

        dists = cdist(gts, pts, metric='euclidean')
        cd = get_cd(dists)
        mcd.append(cd)

        for i in range(11):
            key = i * 0.01
            hiou = hungary_iou(dists, key)
            siou = saliency_iou(dists, key)
            hmiou[key].append(hiou)
            smiou[key].append(siou)

        # continue
        gt_pts = [trimesh.primitives.Sphere(radius=0.02, center=pt).to_mesh() for pt in gts]
        for pt in gt_pts:
            pt.visual.vertex_colors = (255, 0, 0, 255)
        pred_pts = [trimesh.primitives.Sphere(radius=0.02, center=pt).to_mesh() for pt in pts]
        for pt in pred_pts:
            pt.visual.vertex_colors = (0, 255, 0, 255)
        trimesh.Scene([mesh] + gt_pts + pred_pts).show()

    for i in range(11):
        key = i * 0.01
        print(np.mean(smiou[key]))

    for i in range(11):
        key = i * 0.01
        print(np.mean(hmiou[key]))
    print(np.mean(mcd))


def add_noise(x, sigma=0.015, clip=0.05):
    noise = np.clip(sigma*np.random.randn(*x.shape), -clip, clip)
    return x + noise


if __name__ == "__main__":
    mesh_file = 'F:/dataset/Teeth3DS/data/upper/013TXGFK/013TXGFK_upper.obj'
    # mesh_file = 'F:/dataset/3DTeethLand/patches/013TXGFK_upper/11.off'
    json_file = 'F:/dataset/3DTeethLand/Batch_2_4_23_24/013TXGFK_upper__kpt.json'
    # json_file = 'F:/dataset/3DTeethLand/patches/013TXGFK_upper/11.json'

    mesh = trimesh.load(mesh_file)
    annots = json.load(open(json_file))
    landmarks = annots['objects']

    # continue
    mesial_pts = [trimesh.primitives.Sphere(radius=0.2, center=lm['coord']).to_mesh() for lm in landmarks if lm['class'] == 'Cusp']
    for pt in mesial_pts:
        pt.visual.vertex_colors = (255, 0, 0, 255)
    distal_pts = [trimesh.primitives.Sphere(radius=0.2, center=lm['coord']).to_mesh() for lm in landmarks if lm['class'] == 'Distal' or lm['class'] == 'Mesial'
                  or lm['class'] == 'FacialPoint' or lm['class'] == 'InnerPoint' or lm['class'] == 'OuterPoint']
    for pt in distal_pts:
        pt.visual.vertex_colors = (0, 255, 0, 255)
    trimesh.Scene([mesh] + mesial_pts + distal_pts).show()

    print()
