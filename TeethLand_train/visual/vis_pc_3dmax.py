import os
import json
import numpy as np
import trimesh
import matplotlib as mpl
from scipy.spatial.distance import cdist
from utils.data_utils import naive_read_pcd


LABEL2COLOR = {0: "[255,165,0]",#orange
               1: "[0,0,255]",#Blue
               2: "[210,105,30]",#chocolate
               3: "[255,0,255]",#Magenta
               4: "[0,128,128]",#Teal
               5: "[165,42,42]",#brown
               6: "[173,255,47]",#green yellow
               7: "[75,0,130]",#indigo
               8: "[255,105,180]",#hot pink
               9: "[0,255,0]",#Lime
               10: "[143,188,143]",#dark sea green
               11: "[47,79,79]",# dark slate gray
               12: "[128,0,0]",#maroon
               13: "[0,100,0]",#dark green
               14: "[255,255,0]",#yellow
               15: "[220,20,60]",#crimson
               16: "[0,0,128]",#Navy
               17: "[188,143,143]",#rosy brown
               18: "[106,90,205]", #slate blue
               19: "[128,128,0]",#Olive
               20: "[0,128,0]",#Green
               }


#  灰色点云
def get_raw(pcd_file, write_file):
    # mesh
    # mesh = trimesh.load(off_file)
    # v = np.array(mesh.vertices, dtype='float32')
    v = naive_read_pcd(pcd_file)[0]

    note = open(write_file, mode='w')
    for i, vi in enumerate(v):
        str = 'sphere radius:0.005 pos:[%.10f,%.10f,%.10f] wirecolor:[169, 169,169] \n' % (vi[0], vi[1], vi[2])
        note.writelines(str)
    note.close()


def get_gt_saliency(pcd_file, write_file):
    class_id = os.path.dirname(pcd_file).split('/')[-1]
    model_id = os.path.basename(pcd_file).split('.')[0]
    v = naive_read_pcd(pcd_file)[0]
    annots = json.load(open('F:/dataset/keypointnet/annotations/all.json'))
    annots = [annot for annot in annots if annot['class_id'] == class_id]
    keypoints = dict(
        [(annot['model_id'], [kp_info['pcd_info']['point_index'] for kp_info in annot['keypoints']]) for annot in
         annots])
    note = open(write_file, mode='w')
    kp_indices = keypoints[model_id]
    for i, vi in enumerate(v):
        if i in kp_indices:
            str = 'sphere radius:0.02 pos:[%.10f,%.10f,%.10f] wirecolor:[255, 0, 0] \n' % (vi[0], vi[1], vi[2])
        else:
            str = 'sphere radius:0.005 pos:[%.10f,%.10f,%.10f] wirecolor:[169, 169,169] \n' % (vi[0], vi[1], vi[2])
        note.writelines(str)


def get_result_saliency(pcd_file, heat_file, idx_file, write_file):
    v = naive_read_pcd(pcd_file)[0]
    heats = np.loadtxt(heat_file, delimiter=',')
    idx = np.loadtxt(idx_file, delimiter=',', dtype='int64')
    heats = heats[idx]
    h_idx = np.argmax(heats, axis=1)  # B,M
    note = open(write_file, mode='w')
    for i, vi in enumerate(v):
        if i in h_idx:
            str = 'sphere radius:0.02 pos:[%.10f,%.10f,%.10f] wirecolor:[0, 0,255] \n' % (vi[0], vi[1], vi[2])
        else:
            str = 'sphere radius:0.005 pos:[%.10f,%.10f,%.10f] wirecolor:[169, 169,169] \n' % (vi[0], vi[1], vi[2])
        note.writelines(str)
    note.close()


def get_gt_corres(pcd_file, write_file):
    class_id = os.path.dirname(pcd_file).split('/')[-1]
    model_id = os.path.basename(pcd_file).split('.')[0]
    annots = json.load(open('F:/dataset/keypointnet/annotations/all.json'))
    annots = [annot for annot in annots if annot['class_id'] == class_id]
    keypoints = dict([(annot['model_id'],
                       [(kp_info['pcd_info']['point_index'], kp_info['semantic_id']) for kp_info in annot['keypoints']])
                      for annot in annots])
    idx2semid = dict()
    nclasses = max([max([kp_info['semantic_id'] for kp_info in annot['keypoints']]) for annot in annots]) + 1
    curr_keypoints = -np.ones((nclasses,), dtype='int64')
    for i, kp in enumerate(keypoints[model_id]):
        curr_keypoints[kp[1]] = kp[0]
        idx2semid[i] = kp[1]

    v = naive_read_pcd(pcd_file)[0]
    gts = np.array([v[idx] for idx in curr_keypoints if idx >= 0])
    g_idx = np.array([idx for idx in curr_keypoints if idx >= 0])
    sem_labels = list(idx2semid.values())
    sem_labels = np.array(sem_labels, dtype='int64')

    note = open(write_file, mode='w')
    for i, vi in enumerate(v):
        if i in curr_keypoints:
            idx = np.argwhere(g_idx == i)
            l = int(sem_labels[idx])
            color = LABEL2COLOR[l]
            str = 'sphere radius:0.02 pos:[%.10f,%.10f,%.10f] wirecolor:%s \n' % (vi[0], vi[1], vi[2], color)
        else:
            str = 'sphere radius:0.005 pos:[%.10f,%.10f,%.10f] wirecolor:[169, 169,169] \n' % (vi[0], vi[1], vi[2])
        note.writelines(str)
    note.close()


def get_result_corres(pcd_file, heat_file, idx_file, write_file):  # 用saliency的结果对应标签
    class_id = os.path.dirname(pcd_file).split('/')[-1]
    model_id = os.path.basename(pcd_file).split('.')[0]
    annots = json.load(open('F:/dataset/keypointnet/annotations/all.json'))
    annots = [annot for annot in annots if annot['class_id'] == class_id]
    keypoints = dict([(annot['model_id'], [(kp_info['pcd_info']['point_index'], kp_info['semantic_id']) for kp_info in annot['keypoints']]) for annot in annots])
    idx2semid = dict()
    nclasses = max([max([kp_info['semantic_id'] for kp_info in annot['keypoints']]) for annot in annots]) + 1
    curr_keypoints = -np.ones((nclasses,), dtype='int64')
    for i, kp in enumerate(keypoints[model_id]):
        curr_keypoints[kp[1]] = kp[0]
        idx2semid[i] = kp[1]

    v = naive_read_pcd(pcd_file)[0]
    gts = np.array([v[idx] for idx in curr_keypoints if idx >= 0])
    sem_labels = list(idx2semid.values())
    sem_labels = np.array(sem_labels, dtype='int64')

    heats = np.loadtxt(heat_file, delimiter=',')
    idx = np.loadtxt(idx_file, delimiter=',', dtype='int64')
    heats = heats[idx]
    h_idx = np.argmax(heats, axis=1)
    pts = v[h_idx]
    dists = cdist(gts, pts, metric='euclidean')
    p_labels = np.argmin(dists.T, axis=1)
    psem_labels = sem_labels[p_labels]

    note = open(write_file, mode='w')
    for i, vi in enumerate(v):
        if i in h_idx:
            idx = np.argwhere(h_idx == i)
            l = int(psem_labels[idx])
            color = LABEL2COLOR[l]
            str = 'sphere radius:0.02 pos:[%.10f,%.10f,%.10f] wirecolor:%s \n' % (vi[0], vi[1], vi[2], color)
        else:
            str = 'sphere radius:0.005 pos:[%.10f,%.10f,%.10f] wirecolor:[169, 169,169] \n' % (vi[0], vi[1], vi[2])
        note.writelines(str)
    note.close()


def get_pred_corres(pcd_file, heat_file, idx_file, write_file):  # xyz_file label_file

    v = naive_read_pcd(pcd_file)[0]

    xyzs = np.loadtxt(heat_file, delimiter=',')
    labels = np.loadtxt(idx_file, delimiter=',', dtype='int64')

    note = open(write_file, mode='w')
    for i, vi in enumerate(v):
        str = 'sphere radius:0.005 pos:[%.10f,%.10f,%.10f] wirecolor:[169, 169,169] \n' % (vi[0], vi[1], vi[2])
        note.writelines(str)
    for i, vi in enumerate(xyzs):
        l = labels[i]
        color = LABEL2COLOR[l]
        str = 'sphere radius:0.02 pos:[%.10f,%.10f,%.10f] wirecolor:%s \n' % (vi[0], vi[1], vi[2], color)
        note.writelines(str)
    note.close()


def get_compare_corres(pcd_file, idx_file, write_file):  # xyz_file label_file

    v = naive_read_pcd(pcd_file)[0]

    idx = np.loadtxt(idx_file, delimiter=',', dtype='int64')

    note = open(write_file, mode='w')
    for i, vi in enumerate(v):
        str = 'sphere radius:0.005 pos:[%.10f,%.10f,%.10f] wirecolor:[169, 169,169] \n' % (vi[0], vi[1], vi[2])
        note.writelines(str)
    for i, vi in enumerate(v[idx]):
        l = i
        color = LABEL2COLOR[l]
        str = 'sphere radius:0.02 pos:[%.10f,%.10f,%.10f] wirecolor:%s \n' % (vi[0], vi[1], vi[2], color)
        note.writelines(str)
    note.close()


def get_heatmap(pcd_file, heat_file, write_file):
    v = naive_read_pcd(pcd_file)[0]
    heats = np.loadtxt(heat_file, delimiter=',')
    heat = heats[17]
    heat = (heat - np.min(heat)) / (np.max(heat) - np.min(heat))
    note = open(write_file, mode='w')
    cmap = mpl.colormaps['Spectral']
    for i, d in enumerate(heat):
        vi = v[i]
        value = cmap(d)
        str = 'sphere radius:0.005 pos:[%.10f,%.10f,%.10f] wirecolor:[%.10f,%.10f,%.10f] \n' % (vi[0], vi[1], vi[2], value[0]*255, value[1]*255, value[2]*255)
        # str_l.append(str)
        note.writelines(str)
    note.close()


if __name__ == '__main__':
    pcd_file = 'F:/dataset/keypointnet/pcds/03001627/4f1f4c4d1b2f6ac3cf004563556ddb36.pcd'
    heat_file = 'F:/dataset/keypointnet/results/chair/corres_xyz/4f1f4c4d1b2f6ac3cf004563556ddb36.txt'
    idx_file = 'F:/dataset/keypointnet/results/chair/corres_labels/4f1f4c4d1b2f6ac3cf004563556ddb36.txt'
    write_file = 'C:/Users/Ricky/Desktop/keypoint/failure/gt_3.ms'
    # get_result(pcd_file, heat_file, idx_file, write_file)
    # get_raw(pcd_file, write_file)
    # get_result_corres(pcd_file, heat_file, idx_file, write_file)
    # get_pred_corres(pcd_file, heat_file, idx_file, write_file)
    # get_result_saliency(pcd_file, heat_file, idx_file, write_file)
    # get_gt_saliency(pcd_file, write_file)
    get_gt_corres(pcd_file, write_file)

    # label_file = 'F:/dataset/keypointnet/results/compare_results/rsnet/chair/29f890e465741b7ef8cb9d3fa2bcdc0.txt'
    # get_compare_corres(pcd_file, label_file, write_file)

