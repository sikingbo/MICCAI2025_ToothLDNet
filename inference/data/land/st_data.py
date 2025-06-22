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
from data.land.common import calc_features, geodesic_heatmaps, sample
from utils.land.data_utils import geodesic_heatmaps, augment


class TeethLandDataset(Dataset):
    def __init__(self, args, split_file: str, train: bool):
        self.args = args
        self.files = []
        self.num_points = args.num_points
        self.augmentation = args.augmentation if train else False

        with open(os.path.join(args.split_root, split_file)) as f:
            for line in f:
                filename = line.strip()
                teeth_root = os.path.join(self.args.patch_root, filename)

                t_files = os.listdir(teeth_root)
                for f in t_files:
                    if f.endswith('.off'):
                        t_idx = os.path.splitext(f)[0]
                        off_file = os.path.join(teeth_root, t_idx + '.off')
                        js_file = os.path.join(teeth_root, t_idx + '.json')
                        gd_file = os.path.join(teeth_root, t_idx + '_f.npy')
                        if os.path.exists(off_file) and os.path.exists(js_file) and os.path.exists(gd_file):
                            self.files.append((off_file, js_file, gd_file, t_idx))
        random.shuffle(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        off_file, js_file, gd_file, t_idx = self.files[idx]
        mesh = trimesh.load(off_file)
        landmarks = json.load(open(js_file))
        geo_dists = np.load(gd_file)
        heatmaps = geodesic_heatmaps(geo_dists, self.args)

        vs = mesh.vertices
        vs = vs - vs.mean(0)
        fs = mesh.faces
        # if self.augmentation:
        #     vs, fs = augment(vs, fs)
        fs, heatmaps = sample(fs, heatmaps, self.num_points)
        vs = torch.tensor(vs, dtype=torch.float32)
        fs = torch.tensor(fs, dtype=torch.long)
        features = calc_features(vs, fs)  # (15, nf)

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

        return features, int(t_idx), torch.tensor(heatmaps, dtype=torch.float32), torch.tensor(lm_idx, dtype=torch.long)


if __name__ == '__main__':
    class Args(object):
        def __init__(self):
            self.split_root = 'F:/dataset/3DTeethLand'
            self.patch_root = 'F:/dataset/3DTeethLand/patches'
            self.augmentation = True
            self.normalize_pc = True
            self.landmark_std = 0.5
            self.num_points = 10000

    data = TeethLandDataset(Args(), 'train.txt', True)
    i = 0
    for feats, t_idx, heats, labels in data:
        print(t_idx)
        print(feats.shape)
        print(heats.shape)
        # if 0. in labels:
        # if feats.shape[1] < 1000:
        #     print("False")
        i += 1
    print(i)



