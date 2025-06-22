import torch
import numpy as np


def calc_features(vs, ts):
    """
    :param vs: (nv, 3), float
    :param ts: (nf, 3), long
    :return: fea: (15, nf), float
    """
    nf = ts.shape[0]
    fea = torch.empty((15, nf), dtype=torch.float32).to(vs.device)

    vs_in_ts = vs[ts]

    fea[:3, :] = vs_in_ts.mean(1).T[None]  # centers , 3
    fea[3:6, :] = calc_normals(vs, ts).T[None]  # normal, 3
    fea[6:15, :] = (vs_in_ts - vs_in_ts.mean(1, keepdim=True)).reshape((nf, -1)).T[None]
    return fea


def calc_normals(vs: torch.Tensor, ts: torch.Tensor):
    """
    :param vs: (n_v, 3)
    :param ts: (n_f, 3), long
    :return normals: (n_f, 3), float
    """
    normals = torch.cross(vs[ts[:, 1]] - vs[ts[:, 0]],
                          vs[ts[:, 2]] - vs[ts[:, 0]])
    normals /= torch.sum(normals ** 2, 1, keepdims=True) ** 0.5 + 1e-9
    return normals


def Euclidean_heatmaps(pc, kp, args):
    ys = []
    for i, p in enumerate(kp):
        distance = np.linalg.norm(pc - p, axis=-1)
        y = np.exp(- distance ** 2 / (2 * args.landmark_std ** 2))
        ys.append(y)
    return np.asarray(ys).T


def geodesic_heatmaps(geodesic_matrix, args):
    ys = []
    for i, dist in enumerate(geodesic_matrix):
        y = np.exp(- dist ** 2 / (2 * args.landmark_std ** 2))
        ys.append(y)
    return np.asarray(ys)


def sample(mesh, num_points):  # 15xN MxN
    fs = mesh.faces
    # hs = hs.T
    sample_fs = np.zeros((num_points, fs.shape[1]), dtype=int)
    # sample_hs = np.zeros((num_points, hs.shape[1]), dtype=float)

    if len(fs) < num_points:
        mesh = mesh.subdivide_to_size(num_points / len(fs))
        sample_fs = mesh.faces
    else:
        idx = np.random.permutation(len(fs))[:num_points]
        sample_fs[:] = fs[idx]
        # sample_hs[:] = hs[idx]

    return sample_fs  # , sample_hs.T


