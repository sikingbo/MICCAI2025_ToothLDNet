import trimesh
import json
import numpy as np

# visualize triangles with different colors according to their labels.
COLORS = [[185/255,185/255,185/255],               # reserved
          [1.0, 0.7, 0],    # yellow
          [0.9, 0.7, 0.6], # pink
          [0.6, 0.9, 0.5], # green
          [0.5, 0.7, 0.9], # blue
          [0.9, 0.6, 0.9], # purple
          [0, 0.8, 0.9],
          [0.6, 0.4, 0.4],
          [0.6, 0.7, 0.1],
          [0.9, 0.9, 0.3],
          [0.3, 0.6, 0.5],
          [0.8, 0.5, 1.0],
          [0.6, 1.0, 1.0],
          [1.0, 0.6, 0.6],
          [0.0, 0.8, 0.4],
          [1.0, 0.4, 0.7],
          [1.0, 0.8, 0.6]
          ]


def trans_labels(labels):
    tar_labels = [0, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28]  # upper
    # tar_labels = [0, 31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48]  # lower
    tar_labels = np.array(tar_labels)
    for i, l in enumerate(labels):
        idx = np.argwhere(tar_labels == l)
        labels[i] = idx
    return labels


def get_segmentation(off_file, json_file, write_file):
    mesh = trimesh.load(off_file)
    with open(json_file, 'r') as fp:
        json_data = json.load(fp)
    labels = json_data['labels']
    labels = np.array(labels, dtype='int64')
    labels = trans_labels(labels)

    vertex_colors = np.array([COLORS[l] for l in labels])
    mesh.visual.vertex_colors = vertex_colors
    mesh.export(write_file)  # 导出为PLY格式，支持颜色


if __name__ == '__main__':
    # off_file = 'F:/dataset/3DTeethLand/patches/ZKJEPFDD_lower/13.off'
    off_file = 'C:/Users/Ricky/Desktop/teaser/01A6G_upper/01A6GW4A_upper.obj'
    json_file = 'C:/Users/Ricky/Desktop/teaser/01A6G_upper/01A6GW4A_upper.json'
    dist_file = 'F:/dataset/3DTeethLand/patches/ZKJEPFDD_lower/13_f.npy'
    write_file = 'C:/Users/Ricky/Desktop/teaser/01A6G_upper/01A6GW4A_upper_seg.obj'

    get_segmentation(off_file, json_file, write_file)

    # mesh = trimesh.load('C:/Users/Ricky/Desktop/teaser/01A6G_upper/01A6GW4A_upper.obj')
    # mesh.export('C:/Users/Ricky/Desktop/teaser/01A6G_upper/01A6GW4A_upper.off')




