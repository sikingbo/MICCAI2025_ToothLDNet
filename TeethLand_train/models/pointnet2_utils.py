import torch


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def index_points_group(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S1, S2]
    Return:
        new_points: indexed points data, [B, S1, S2, C]
    """
    device = points.device
    B, N, C = points.shape
    S1, S2 = idx.shape[1], idx.shape[2]

    view_shape = [B, 1, 1]
    repeat_shape = [1, S1, S2]

    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]

    return new_points



def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def fps_subsample(xyz, n_points=2048):
    """
    Args
        pcd: (b, 16384, 3)

    returns
        new_pcd: (b, n_points, 3)
    """
    new_xyz = index_points(xyz, farthest_point_sample(xyz, n_points))
    return new_xyz #.squeeze().cpu().numpy()


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def query_knn(k, xyz, new_xyz, include_self=True):
    """Find k-NN of new_xyz in xyz"""
    pad = 0 if include_self else 1
    sqr_dists = square_distance(new_xyz, xyz)  # B, S, N
    _, idx = torch.sort(sqr_dists, dim=-1, descending=False)
    idx = idx[:, :, pad: k+pad]
    return idx.long()


def sample_and_group_knn(xyz, feats, npoint, k):
    """
    Args:
        xyz: Tensor, (B, 3, N) coordinats
        feats: Tensor, (B, f, N) features
        npoint: int sampled n points
        k: int

    Returns:
        new_xyz: Tensor, (B, 3, npoint) sampled points coordinates
        new_feats: Tensor, (B, 3 | f+3 | f, npoint, nsample)
        idx_local: Tensor, (B, npoint, nsample)
        grouped_xyz: Tensor, (B, 3, npoint, nsample)
    """

    xyz_flipped = xyz.permute(0, 2, 1).contiguous()  # (B, N, 3)
    B, N, C = xyz_flipped.shape
    S = npoint
    new_xyz = index_points(xyz_flipped, farthest_point_sample(xyz_flipped, npoint))  # (B, npoint, 3)
    grouped_idx = query_knn(k, xyz_flipped, new_xyz)

    grouped_xyz = index_points(xyz_flipped, grouped_idx)  # [B, npoint, k, 3]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)  # [B, npoint, k, 3]

    grouped_feats = index_points(feats.permute(0, 2, 1).contiguous(), grouped_idx)  # [B, npoint, k, f]
    # grouped_feats = torch.cat([grouped_xyz_norm, grouped_feats], dim=-1)  # [B, npoint, nsample, 3+f]

    return new_xyz, grouped_feats, grouped_xyz, grouped_idx


def normalized(tensor):
    min_val = tensor.min(dim=2, keepdim=True)[0]
    max_val = tensor.max(dim=2, keepdim=True)[0]
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor