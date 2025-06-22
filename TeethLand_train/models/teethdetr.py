import torch
import torch.nn as nn
from .dgcnn_point import Backbone_point, SharedMLP1d, SemanticBranch_point
from .dgcnn_global import Backbone_global, SemanticBranch_global
from .pointnet2_utils import index_points
from .transformer import TransformerFeatureEnhancer
import numpy as np
import time
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances_argmin_min


class OffsetBranch(nn.Module):
    def __init__(self, args):
        super(OffsetBranch, self).__init__()

        self.smlp1 = SharedMLP1d([args.emb_dims+args.n_edgeconvs_backbone*64, 256, 128], args.norm)
        self.dp = nn.Dropout1d(args.dropout)
        self.smlp2 = nn.Conv1d(128, 3, kernel_size=1)

    def forward(self, x):
        return self.smlp2(self.dp(self.smlp1(x)))


from sklearn.cluster import DBSCAN, KMeans
def split_clusters(X, clusters):
    from sklearn.decomposition import PCA
    new_clusters = []
    for c in clusters:
        x = X[c, :2]
        x = PCA(2).fit_transform(x)

        length = x.max(0)-x.min(0)
        if np.any(length>10):
            # split
            kmeans = KMeans(n_clusters=2, random_state=0).fit(x)
            idx = np.where(kmeans.labels_ == 1)[0]
            new_clusters.append(c[idx])
            idx = np.where(kmeans.labels_ == 0)[0]
            new_clusters.append(c[idx])
        else:
            new_clusters.append(c)
    return new_clusters


def adjust_tensor(tensor, target_length=512):
    # 获取当前张量的大小
    B, C = tensor.shape  # B: batch size, C: feature dimension (c)
    # 如果 c 大于 500，随机采样
    if C > target_length:
        # 随机选择 target_length 个索引
        idx = torch.randint(0, C, (target_length,))
        sampled_tensor = tensor[:, idx]  # 根据索引采样
        return sampled_tensor
    # 如果 c 小于 500，进行重复
    elif C < target_length:
        # 计算需要重复多少次
        repeat_count = target_length // C + 1  # 多重复几次以确保足够
        repeated_tensor = tensor.repeat(1, repeat_count)[:, :target_length]  # 只取前 target_length 个元素
        return repeated_tensor
    # 如果 c 恰好等于 500，直接返回原张量
    return tensor


def cluster_faces(feat2, p, offsets, vs_mean, device):
    B, N, _ = p.shape  # B: batch size, N: number of faces
    # 筛选掉偏移量小于阈值的点
    thing_idx = np.arange(feat2.shape[2])
    # thing_idx = np.where(torch.sum(offsets ** 2, dim=2).detach().cpu().numpy()**0.5 > 0.01)[0]
    feat2 = feat2[:, :, thing_idx]
    # 将 p 转换为 numpy 数组，并提取聚类所需的索引
    p_cpu = p.view(-1, 3).cpu().numpy()  # (B*N, 3) 重新调整形状，确保它是一个 2D 数组
    # 使用 DBSCAN 聚类
    # start_time = time.time()  # 记录开始时间
    clustering = DBSCAN(eps=1.05, min_samples=30).fit(p_cpu[thing_idx])  # 仅对有效点进行聚类
    # end_time = time.time()  # 记录结束时间
    # print(f"执行耗时: {end_time - start_time:.4f} 秒")  # 保留4位小数
    labels = clustering.labels_

    # 获取每个聚类的点
    unique_labels = np.unique(labels)
    clusters = [thing_idx[labels == label] for label in unique_labels if label != -1]
    # 如果需要，拆分聚类（可取消）
    # clusters = split_clusters(p_cpu, clusters)
    # 计算每个聚类的质心
    cluster_centers = np.asarray([np.mean(p_cpu[c], axis=0) for c in clusters])

    # 将聚类质心转换回 Tensor
    cluster_centers_tensor = torch.tensor(cluster_centers, device=device)  # (T, 3)
    if len(cluster_centers_tensor)==0:
        return torch.zeros(B, feat2.shape[1], 1), -1
    # 提取聚类后的特征
    clustered_features_list = []
    for b in range(B):
        # 针对每个聚类，计算该聚类的平均特征
        batch_clustered_features = []
        for c in clusters:
            # if len(c) < 75: continue
            # 仅选择当前聚类的面
            cluster_features = feat2[b, :, c]  # 形状为 (128, T)
            # 计算当前聚类的平均特征
            avg_clustered_feature = torch.mean(cluster_features, dim=-1)  # 计算每个聚类的平均特征
            batch_clustered_features.append(avg_clustered_feature)
        # 将每个聚类的平均特征合并
        if len(batch_clustered_features) > 0:
            clustered_features_list.append(torch.stack(batch_clustered_features, dim=-1))  # (128, T)
    # 聚类特征转换为 Tensor
    if len(clustered_features_list) == 0:
        return torch.zeros(B, feat2.shape[1], 1), -1
    clustered_features_tensor = torch.stack(clustered_features_list, dim=0).to(device)  # (B, 128, T)

    # 计算每个聚类的中心到牙齿中心的距离
    closest_cluster_indices = []
    for b in range(B):
        distances_to_vs = cdist(cluster_centers_tensor.cpu().numpy(), vs_mean[b].cpu().numpy().reshape(1, -1))
        closest_cluster_idx = np.argmin(distances_to_vs)
        closest_cluster_indices.append(closest_cluster_idx)

    # 返回聚类后的特征和距离牙齿中心最近的聚类的索引
    return clustered_features_tensor, torch.tensor(closest_cluster_indices, device=device)


def sample(clustered_features, num, idx):
    B, C, T = clustered_features.shape
    result = torch.zeros(B, C, num)
    if T > num:
        t = clustered_features[:, :, idx].squeeze(-1)
        result[:, :, 0] = clustered_features[:, :, idx].squeeze(-1)
        clustered_features = torch.cat([clustered_features[:, :, :idx], clustered_features[:, :, idx + 1:]], dim=2)
        result[:, :, 1:] = clustered_features[:, :, :num-1]
        idx = 0
    elif T < num:
        result[:, :, :T] = clustered_features
        result[:, :, T:] = clustered_features[:, :, 0].unsqueeze(-1).expand(B, C, num-T)
    else:
        result = clustered_features
    return result, idx


class MLPFusion(nn.Module):
    def __init__(self, input_dim, cluster_dim, hidden_dim):
        super(MLPFusion, self).__init__()
        self.fc1 = nn.Linear(input_dim + cluster_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)  # 输出和原始特征维度相同

    def forward(self, features, cluster_features):
        B, D, N = features.shape
        cluster_features = cluster_features.unsqueeze(-1).expand(-1, -1, N)  # B x D x N
        # 拼接特征：B x (D + cluster_dim) x N
        combined_features = torch.cat([features, cluster_features], dim=1)
        # 通过全连接层
        combined_features = combined_features.permute(0, 2, 1)  # 转换为 B x N x (D + cluster_dim)
        combined_features = self.fc1(combined_features)  # B x N x hidden_dim
        combined_features = self.fc2(combined_features)  # B x N x D

        return combined_features.permute(0, 2, 1)  # B x 128 x N


class TeethDETR(nn.Module):
    """
    @param args: k, number of neighbors
                input_channels, int
                output_channels, int
                dynamic, bool, if using dynamic or not.
                transform, bool, if using the transform net
                n_edgeconvs_backbone, int, the number of EdgeConvs in the backbone
                emb_dims, int
                global_pool_backbone, str, "avg" or "max"
                norm, str, "instance" or "batch"
                dropout, float
    """
    def __init__(self, args):
        super(TeethDETR, self).__init__()

        self.query_num = args.query_num
        self.num_clu = args.num_clu
        # self.n_decoder = args.n_decoder
        self.dynamic = args.dynamic
        self.backbone_point = Backbone_point(args)
        self.backbone_global = Backbone_global(args)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.semantic_branch_point = SemanticBranch_point(args)
        self.semantic_branch_global = SemanticBranch_global(args)
        self.offset_branch = OffsetBranch(args)
        self.transformer_encoder = TransformerFeatureEnhancer(d_model=128, nhead=8, num_layers=4)
        self.merge = MLPFusion(input_dim=128, cluster_dim=128, hidden_dim=256)

        self.heat_decoder = nn.Sequential(SharedMLP1d([128, 64], args.norm),
                                          nn.Dropout1d(args.dropout),
                                          nn.Conv1d(64, args.query_num, kernel_size=1))
        self.decoder_1 = SharedMLP1d([args.num_points, 1024], args.norm)
        self.decoder_2 = nn.Sequential(SharedMLP1d([1024, 256], args.norm),
                                       SharedMLP1d([256, 128], args.norm))

        self.prob_out = nn.Sequential(SharedMLP1d([128, 64], args.norm),
                                      nn.Dropout(args.dropout),
                                      nn.Conv1d(64, 7, kernel_size=1),)
                                      # nn.Sigmoid())
        self.heat_out = nn.Sequential(SharedMLP1d([args.query_num, args.query_num], args.norm),
                                      nn.Dropout(args.dropout),
                                      nn.Conv1d(args.query_num, args.query_num, kernel_size=1))

    def forward(self, features1, features2, vs_mean):
        device = features1.device

        xyz = features1[:, :3, :].contiguous()  # B,3,N
        feat1, _ = self.backbone_point(features1)
        feat1 = self.semantic_branch_point(feat1, feat1 if self.dynamic else xyz)  # B,128,N

        p = features2[:, :3, :].contiguous()
        feat2 = self.backbone_global(features2)
        offsets = self.offset_branch(feat2)
        p = p + offsets.detach()
        feat2 = self.semantic_branch_global(feat2, p)  # B,128,N2
        clustered_features0, closest_cluster_idx = cluster_faces(feat2, p.permute(0, 2, 1), offsets.permute(0, 2, 1), vs_mean, device=device)  # B,128,T
        if clustered_features0.shape[2] > 6 and closest_cluster_idx >= 0:
            clustered_features, closest_cluster_idx = sample(clustered_features0, self.num_clu, closest_cluster_idx)
            clustered_features = clustered_features.to(device)
            clustered_features = self.transformer_encoder(clustered_features)  # B,128,N2
            feat2 = clustered_features[:, :, closest_cluster_idx].squeeze(-1)
            feat = self.merge(feat1, feat2)
        else:
            feat = feat1
        # feat  B,128,N
        heat_feat = self.heat_decoder(feat)  # B,M,N
        query_feat = self.decoder_1(heat_feat.permute(0, 2, 1))  # B,1024,M
        query_feat = self.decoder_2(query_feat)  # B, 128, M

        heat = self.heat_out(heat_feat)
        prob = self.prob_out(query_feat)  # B,1,M

        return prob.permute(0, 2, 1), heat, offsets

    def inference(self, x, gt):
        xyz = x[:, :3, :].contiguous()  # B,3,N
        xyz_t = xyz.permute(0,2,1)

        feat,_ = self.backbone(x)
        feat = self.semantic_branch(feat, feat if self.dynamic else xyz)  # B,128,N

        heat_feat = self.heat_decoder(feat)  # B,M,N
        query_feat = self.decoder_1(heat_feat.permute(0, 2, 1))  # B,1024,M
        query_feat = self.decoder_2(query_feat)  # B, 128, M

        heat = self.heat_out(heat_feat)
        prob = self.prob_out(query_feat)  # B,1,M

        pred_class = torch.argmax(prob, dim=1).squeeze()  # 预测的所有query类别
        h_idx = torch.argmax(heat, dim=2)  # B,M
        p_xyz = index_points(xyz_t, h_idx)

        indices = torch.nonzero(pred_class != 0, as_tuple=False).squeeze()
        p_xyz = p_xyz[:, indices, :]

        return p_xyz.squeeze(), pred_class
