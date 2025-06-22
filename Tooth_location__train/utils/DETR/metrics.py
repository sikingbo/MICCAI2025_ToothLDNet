import numpy as np
from scipy.optimize import linear_sum_assignment


def pairwise_distance(pts1, pts2):
    """
    计算两组关键点之间的距离矩阵
    """
    return np.sqrt(np.sum((pts1[:, None] - pts2) ** 2, axis=-1))


def calculate_iou(gt_pts, pred_pts, threshold):
    """
    计算关键点之间的IoU
    """
    gt_pts = np.array(gt_pts)
    pred_pts = np.array(pred_pts)

    distance_matrix = pairwise_distance(pred_pts, gt_pts)
    matches = distance_matrix <= threshold

    intersection = np.sum(matches, axis=1)
    union = len(gt_pts) + len(pred_pts) - intersection

    iou = intersection / union
    return iou


def calculate_miou(gt_pts, pred_pts, threshold=0.1):
    """
    计算关键点之间的mIoU
    """
    iou_values = calculate_iou(gt_pts, pred_pts, threshold)
    miou = np.mean(iou_values)
    return miou


def get_iou(lg, lp, dists, threshold=0.1):
    c = np.sum(dists < threshold)
    iou = c/(lg + lp - c)
    return iou


# 最大包围框边长
def get_len(xyz):
    # 计算最大包围框的边长
    min_x, min_y, min_z = np.min(xyz, axis=0)
    max_x, max_y, max_z = np.max(xyz, axis=0)
    length_x = max_x - min_x
    length_y = max_y - min_y
    length_z = max_z - min_z

    # 最大包围框的边长为三个维度上的差值
    max_length = max(length_x, length_y, length_z)

    return max_length


# def eval_iou(dists, dist_thresh=0.1):
#     gt_l = dists.shape[0]
#     pred_l = dists.shape[1]
#
#     fp = np.count_nonzero(np.all(dists.T > dist_thresh, axis=-1))
#     fn = np.count_nonzero(np.all(dists > dist_thresh, axis=-1))
#
#     return (gt_l - fn) / np.maximum(gt_l + fp, np.finfo(np.float64).eps)


def eval_iou(dists, dist_thresh=0.1):

    gt_l = dists.shape[0]
    pred_l = dists.shape[1]

    fp = np.count_nonzero(np.all(dists.T > dist_thresh, axis=-1))
    fn = np.count_nonzero(np.all(dists > dist_thresh, axis=-1))
    tp = gt_l - fn

    # return (gt_l - fn) / np.maximum(gt_l + fp, np.finfo(np.float64).eps)
    return tp / (gt_l + pred_l - tp)


def saliency_iou(dists, dist_thresh=0.1):
    distances = dists.copy()
    gt_l = distances.shape[0]
    pred_l = distances.shape[1]

    tp = 0
    for i in range(gt_l):
        ds = distances[i]
        idx = np.argmin(ds)
        d_min = np.min(ds)
        if d_min < dist_thresh + 0.001:
            tp += 1
            distances[:, idx] = 1
    return tp / (gt_l + pred_l - tp)


def hungary_iou(dists, dist_thresh=0.1):
    distances = dists.copy()
    gt_l = distances.shape[0]
    pred_l = distances.shape[1]
    indice = linear_sum_assignment(distances)
    distances = distances[indice[0], indice[1]]
    tp = np.sum(distances < dist_thresh + 0.001)
    return tp / (gt_l + pred_l - tp)


def eval_pck(p_l, dists, dist_thresh=0.1):

    fp = np.count_nonzero(np.all(dists.T > dist_thresh, axis=-1))

    return (p_l - fp) / p_l


def chamfer_distance(points1, points2):
    # 计算每个点到另一组点云的最近邻距离
    dist1 = np.sqrt(np.sum((points1[:, None, :] - points2[None, :, :]) ** 2, axis=-1))
    dist2 = np.sqrt(np.sum((points2[:, None, :] - points1[None, :, :]) ** 2, axis=-1))

    # 计算每个点到另一组点云的最近邻距离之和
    cd = np.sum(np.min(dist1, axis=1)) + np.sum(np.min(dist2, axis=1))

    # 计算平均 Chamfer Distance
    avg_cd = cd / (points1.shape[0] + points2.shape[0])

    return avg_cd


def get_cd(dists):
    cd = np.sum(np.min(dists, axis=1)) / dists.shape[0] + np.sum(np.min(dists.T, axis=1)) / dists.shape[1]
    return cd


