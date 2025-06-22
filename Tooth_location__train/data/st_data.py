import os
import numpy as np
import random
import json

import torch
import trimesh
from torch.utils.data import Dataset

from data.common import calc_features
from utils.TeethGNN.data_utils import augment, get_offsets


class Teeth3DS(Dataset):
    def __init__(self, args, split_file: str, train: bool):
        # args
        self.args = args
        self.root_dir = args.root_dir
        self.num_points = args.num_points
        # self.tar_labels = args.tar_labels
        self.augmentation = args.augmentation if train else False
        # files
        self.files = []
        # i=0
        args.split_dir = 'E:/dataset/Teeth_box'
        self.root_dir = 'E:/dataset/Teeth_box/data'
        with open(os.path.join(args.split_dir, split_file)) as f:
            for line in f:
                # i=i+1
                # print(i)
                # print(line)
                filename = line.strip().split('_')[0]
                category = line.strip().split('_')[1]
                teeth = filename + '_' + category
                # print(teeth)
                root = os.path.join(self.root_dir, teeth)
                teeth_ori_file = os.path.join(root, f'{teeth}.obj')  # sim_off
                obj_file = os.path.join(root, f'{line.strip()}_sim.off')
                obj_translation_file = os.path.join(root, 'box_augment', 'translation.obj')
                obj_rotation_file = os.path.join(root, 'box_augment', 'rotation.obj')
                obj_scaling_file = os.path.join(root, 'box_augment', 'scaling.obj')

                txt_file = os.path.join(root, f'{line.strip()}_sim_box.txt')
                txt_gc_file = os.path.join(root, f'{line.strip()}_sim_box_gc.txt')
                if os.path.exists(obj_file) and os.path.exists(txt_file):
                    self.files.append((obj_file, txt_file))
                if os.path.exists(obj_file) and os.path.exists(txt_gc_file):
                    self.files.append((obj_file, txt_gc_file))
                    self.files.append((obj_translation_file, txt_gc_file))
                    self.files.append((obj_rotation_file, txt_gc_file))
                    self.files.append((obj_scaling_file, txt_gc_file))
        random.shuffle(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        obj_file, txt_file = self.files[idx]

        mesh = trimesh.load(obj_file)
        vs, fs = mesh.vertices, mesh.faces
        cs = mesh.triangles_center
        labels = np.loadtxt(txt_file, dtype=np.int32)

        # augmentation
        if self.augmentation:
            vs, fs = augment(vs, fs)
        # sample
        # _, fids = trimesh.sample.sample_surface_even(mesh, self.num_points)
        fids = np.arange(self.num_points)
        fs, labels = fs[fids], labels[:, fids]
        cs = cs[fids]
        # extract input features
        vs = torch.tensor(vs, dtype=torch.float32)
        fs = torch.tensor(fs, dtype=torch.long)
        features = calc_features(vs, fs)   # (15, nf)
        # labels = [np.argwhere(l == self.tar_labels) for l in labels]
        labels = np.array(labels, dtype='float64').squeeze()
        offsets = get_offsets(cs, labels)
        labels_hot = np.zeros((labels.shape[0], 2, labels.shape[1]))

        # 2. 根据 labels 向量填充矩阵
        for i in range(labels.shape[0]):
            labels_hot[i, 0, labels[i] == 0] = 1
            labels_hot[i, 1, labels[i] == 1] = 1

        # return features, torch.tensor(labels, dtype=torch.long), torch.tensor(offsets.T, dtype=torch.float32)
        return features, torch.tensor(labels_hot, dtype=torch.float32), torch.tensor(offsets.T, dtype=torch.float32)


if __name__ == '__main__':
    class Args(object):
        def __init__(self):
            self.root_dir = 'D:/dataset/Teeth3DS/data'
            self.split_dir = 'D:/dataset/Teeth3DS/split'
            self.num_points = 8000
            self.augmentation = True

    data = Teeth3DS(Args(), 'training_all.txt', True)
    i = 0
    for f, l, o in data:
        print(f.shape)
        print(l.shape)
        print(o.shape)
        i += 1

    print(i)