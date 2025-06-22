import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


# prob: b, num_query, 1
# out: b, num_query, num_points/3
# tgt: b, num_landmarks, num_points/3
class HungarianMatcher(nn.Module):
    def __init__(self, cost_prob: float = 1, cost_heat: float = 1):
        super().__init__()
        self.cost_prob = cost_prob
        self.cost_heat = cost_heat
        assert cost_prob != 0 or cost_heat != 0 , "all costs cant be 0"

    @torch.no_grad()
    def forward(self, prob, label, heat, gt):
        bs, num_query = heat.shape[:2]

        heat_y = heat.flatten(0, 1)  # [batch_size * num_queries, n/3]
        gt_y = gt.flatten(0, 1)  # [batch_size * num_queries, m/3]
        cost_h = torch.cdist(heat_y, gt_y, p=1)

        prob_y = prob.flatten(0, 1)
        label_y = label.flatten(0, 1)
        cost_prob = 1 - prob_y[:, label_y]
        # cost_prob = prob_y.expand(prob_y.shape[0], cost_h.shape[1])

        # Final cost matrix
        C = self.cost_heat * cost_h + self.cost_prob * cost_prob
        C = C.view(bs, num_query, -1).cpu()

        sizes = [len(v) for v in gt]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]