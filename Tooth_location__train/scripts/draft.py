import os
import trimesh
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

split_dir = 'D:/dataset/Teeth_box'
root_dir = 'D:/dataset/Teeth_box/data'
m = 100000
i = 0
with open(os.path.join(split_dir, 'all.txt')) as f:
    for line in f:
        i = i + 1
        filename = line.strip().split('_')[0]
        category = line.strip().split('_')[1]
        teeth = filename + '_' + category
        root = os.path.join(root_dir, teeth)
        teeth_sim_file = os.path.join(root, f'{teeth}_sim.off')  # sim_off
        mapping_file = os.path.join(root, f'{teeth}_mapping.npy')  # mapping
        txt_sim_file = os.path.join(root, f'{teeth}_sim_box.txt')  # GNN_gt
        labels = np.loadtxt(txt_sim_file, dtype=np.int32)
        mapping = np.load(mapping_file)
        # mesh = trimesh.load(teeth_sim_file)
        # cs = mesh.triangles_center
        max_value = max(mapping)
        print(f'{i}, box_txt长度：{len(labels)}, mapping最大值：{max_value}')
