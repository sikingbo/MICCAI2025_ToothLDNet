import os
import json
import numpy as np
import trimesh
import matplotlib as mpl
from scipy.spatial.distance import cdist
from utils.data_utils import naive_read_pcd


def get_saliency(pc, p_kps, g_kps):
    write_file = 'C:/Users/Ricky/Desktop/keypoint/noise/pred_020.ms'
    note = open(write_file, mode='w')
    for i, vi in enumerate(pc):
        str = 'sphere radius:0.005 pos:[%.10f,%.10f,%.10f] wirecolor:[169, 169,169] \n' % (vi[0], vi[1], vi[2])
        note.writelines(str)
    for i, vi in enumerate(p_kps):
        str = 'sphere radius:0.02 pos:[%.10f,%.10f,%.10f] wirecolor:[0, 0, 255] \n' % (vi[0], vi[1], vi[2])
        note.writelines(str)
    note.close()

    write_file = 'C:/Users/Ricky/Desktop/keypoint/noise/gt_020.ms'
    note = open(write_file, mode='w')
    for i, vi in enumerate(pc):
        str = 'sphere radius:0.005 pos:[%.10f,%.10f,%.10f] wirecolor:[169, 169,169] \n' % (vi[0], vi[1], vi[2])
        note.writelines(str)
    for i, vi in enumerate(g_kps):
        str = 'sphere radius:0.02 pos:[%.10f,%.10f,%.10f] wirecolor:[255, 0, 0] \n' % (vi[0], vi[1], vi[2])
        note.writelines(str)
    note.close()

