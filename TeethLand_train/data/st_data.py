import os
import numpy as np
import random
import json
from glob import glob
import torch
import torch.nn.functional as F
import trimesh
from scipy.stats import special_ortho_group
from torch.utils.data import Dataset
from data.common import calc_features, geodesic_heatmaps, sample
from utils.data_utils import geodesic_heatmaps, augment, get_offsets


class TeethLandDataset(Dataset):
    def __init__(self, args, split_file: str, train: bool):
        self.args = args
        self.files = []
        self.num_points = args.num_points
        self.num_points_seg = args.num_points_seg
        self.augmentation = args.augmentation if train else False

        with open(os.path.join(args.split_root, split_file)) as f:
            for line in f:
                filename = line.strip()
                teeth_root = os.path.join(self.args.patch_root, filename, 'patches_box')

                t_files = os.listdir(teeth_root)
                for f in t_files:
                    if f.endswith('.off'):
                        t_idx = os.path.splitext(f)[0]
                        off_file = os.path.join(teeth_root, t_idx + '.off')
                        js_file = os.path.join(teeth_root, t_idx + '.json')
                        gd_file = os.path.join(teeth_root, t_idx + '.npy')
                        map_file = os.path.join(teeth_root, f'{t_idx}_map.npy')
                        mesh_file = os.path.join(self.args.patch_root, filename, f'{filename}_sim.off')
                        txt_gc_file = os.path.join(self.args.patch_root, filename, f'{line.strip()}_sim_box_gc.txt')
                        if os.path.exists(off_file) and os.path.exists(js_file) and os.path.exists(gd_file) \
                                and os.path.exists(map_file) and os.path.exists(mesh_file) and os.path.exists(txt_gc_file):
                            self.files.append((off_file, js_file, gd_file, map_file, mesh_file, txt_gc_file))
        random.shuffle(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        off_file, js_file, gd_file, map_file, mesh_file, txt_file= self.files[idx]
        mesh_sim = trimesh.load(mesh_file)
        mesh = trimesh.load(off_file)
        mapping = np.load(map_file)
        landmarks = json.load(open(js_file))
        geo_dists = np.load(gd_file)
        heatmaps = geodesic_heatmaps(geo_dists, self.args)
        labels = np.loadtxt(txt_file, dtype=np.int32)

        vs, fs = mesh.vertices, mesh.faces
        vs_mean = vs.mean(0)
        vs = vs - vs.mean(0)
        if self.augmentation:
            vs, fs = augment(vs, fs)
        # fs, heatmaps = sample(fs, heatmaps, self.num_points)
        _, fids = trimesh.sample.sample_surface_even(mesh, self.num_points)
        fs, heatmaps, mapping = fs[fids], heatmaps[fids], mapping[fids]
        vs = torch.tensor(vs, dtype=torch.float32)
        fs = torch.tensor(fs, dtype=torch.long)
        features1 = calc_features(vs, fs)  # (15, nf)

        vs_sim, fs_sim = mesh_sim.vertices, mesh_sim.faces
        cs_sim = mesh_sim.triangles_center
        if self.augmentation:
            vs_sim, fs_sim = augment(vs_sim, fs_sim)
        fids = np.arange(self.num_points_seg)
        fs_sim, labels, cs_sim = fs_sim[fids], labels[:, fids], cs_sim[fids]
        vs_sim = torch.tensor(vs_sim, dtype=torch.float32)
        fs_sim = torch.tensor(fs_sim, dtype=torch.long)
        features2 = calc_features(vs_sim, fs_sim)   # (15, nf)
        labels = np.array(labels, dtype='float64').squeeze()
        offsets = get_offsets(cs_sim, labels)

        lm_idx = np.zeros(geo_dists.shape[0])
        for i, lm in enumerate(landmarks):
            if lm['class'] == 'Mesial':
                lm_idx[i] = 1
            if lm['class'] == 'Distal':
                lm_idx[i] = 2
            if lm['class'] == 'InnerPoint':
                lm_idx[i] = 3
            if lm['class'] == 'OuterPoint':
                lm_idx[i] = 4
            if lm['class'] == 'FacialPoint':
                lm_idx[i] = 5
            if lm['class'] == 'Cusp':
                lm_idx[i] = 6

        return (features1, features2, torch.tensor(heatmaps.T, dtype=torch.float32), torch.tensor(lm_idx, dtype=torch.long),
                mapping, torch.tensor(offsets.T, dtype=torch.float32), torch.tensor(vs_mean, dtype=torch.float32))


if __name__ == '__main__':
    class Args(object):
        def __init__(self):
            self.split_root = 'F:/dataset/3DTeethLand'
            self.patch_root = 'F:/dataset/3DTeethLand/patches'
            self.augmentation = True
            self.normalize_pc = True
            self.landmark_std = 0.5
            self.num_points = 5000

    data = TeethLandDataset(Args(), 'train.txt', True)
    i = 0
    for feats, tidx, oh,  ch in data:
        print(tidx)
        print(feats.shape)
        print(oh.shape)
        print(ch.shape)
        # if 0. in labels:
        # if feats.shape[1] < 1000:
        #     print("False")
        i += 1
    print(i)



