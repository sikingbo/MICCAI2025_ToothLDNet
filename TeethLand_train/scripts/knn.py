import os
import torch
import trimesh
import numpy as np
import json
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics

tar_labels = [11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28]


# tar_labels = [31,32,33,34,35,36,37,38,41,42,43,44,45,46,47,48]


def trans_labels(labels):
    for i, l in enumerate(tar_labels):
        idx = np.argwhere(labels == l)
        labels[idx] = i + 1
    return labels


def knn(x, k):  # x nx3
    inner = -2 * torch.matmul(x, x.T)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(1, 0)

    _, idx = pairwise_distance.topk(k=k, dim=-1)

    return idx


def knn_map(sim_mesh, ori_mesh, sim_pred):
    sam_c = sim_mesh.triangles_center
    ori_c = ori_mesh.vertices

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(sam_c)
    _, indices = nbrs.kneighbors(ori_c)
    indices = indices.astype(int).flatten()

    ori_pred = sim_pred[indices]
    return ori_pred
    # np.savetxt(ori_txt, ori_pred, fmt='%d', delimiter=',')
    # acc = metrics.accuracy_score(list(ori_pred.squeeze()), list(ori_gt.squeeze()))
    # print(acc)
    # return acc


# ground truth vertices -> faces
def knn_map_gt(mesh_file, gt_json, write_file):
    mesh = trimesh.load(mesh_file)
    cs = mesh.triangles_center
    vs = mesh.vertices

    with open(gt_json, 'r') as fp:
        json_data = json.load(fp)
    labels = json_data['labels']
    labels = np.array(labels, dtype='int64')
    labels = trans_labels(labels)

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(vs)
    _, indices = nbrs.kneighbors(cs)


if __name__ == '__main__':
    sam_file = 'D:/code/teeth/union/data/dataset/data/lower/01A6HAN6/sim/01A6HAN6_lower.obj'
    ori_file = 'D:/code/teeth/union/data/dataset/data/lower/01A6HAN6/01A6HAN6_lower.obj'
    sam_txt = 'D:/code/teeth/union/data/dataset/data/lower/01A6HAN6/sim/01A6HAN6_lower.txt'
    ori_txt = 'D:/code/teeth/union/data/dataset/data/lower/01A6HAN6/01A6HAN6_lower.txt'
    gt_json = 'D:/code/teeth/union/data/dataset/data/lower/01A6HAN6/01A6HAN6_lower.json'
    knn_map(sam_file, ori_file, sam_txt, ori_txt, gt_json)

    '''
    mesh = trimesh.load(ori_file)
    vs = mesh.vertices
    fs = mesh.faces
    print(len(vs))
    print(len(fs))
    '''