import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from .matcher import HungarianMatcher


class Criterion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.matcher = HungarianMatcher(cost_prob=100, cost_heat=1)
        self.loss_h = nn.MSELoss()
        self.loss_dice = DiceLoss()
        # self.loss_focal = FocalLoss(num_classes=2)  # FocalLoss2(alpha=0.25, gamma=2.0, reduction='mean')
        self.loss_c = FocalLoss(num_classes=2)

    def forward(self, g_heats, probs, heats):
        indices = self.matcher(probs, heats, g_heats)

        loss = 0
        for i, idx in enumerate(indices):  # for batch_size
            # for match 0
            gheat = g_heats[i, idx[1], :].squeeze()
            gheat = gheat.view(gheat.size(0), 2, -1)  # K,2,N
            pheat = heats[i, idx[0], :]  #
            pheat = pheat.view(pheat.size(0), 2, -1)  # K,2,N

            gprob = torch.zeros(probs.shape[1], dtype=torch.long).to(probs.device)
            gprob[idx[0]] = 1

            n = pheat.size(0)
            loss_ce = 0
            loss_mse = 0
            for j in range(n):
                pre, gt = pheat[j], gheat[j]  # 2,N
                loss_mse = loss_mse + self.loss_h(pre, gt)
                # loss_dice = loss_dice + self.loss_dice(pre, gt)
                # loss_focal = loss_focal + self.loss_focal(pre, gt)
                target = gt.argmax(dim=0)
                loss_ce = loss_ce + F.cross_entropy(pre.T, target)
                # loss_focal = loss_focal + self.loss_focal(pre.T, target)
            loss_c = self.loss_c(probs.squeeze(), gprob)

            loss = loss + loss_c + loss_mse / n * 5
        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth  # 平滑因子避免除以零

    def forward(self, pre, gt):
        # 假设pre和gt都是2xN的tensor，其中第二维是N个样本
        intersection = torch.sum(pre * gt, dim=1)  # 每一列的交集部分
        union = torch.sum(pre, dim=1) + torch.sum(gt, dim=1)  # 每一列的并集部分
        dice = (2 * intersection + self.smooth) / (union + self.smooth)  # 计算Dice系数
        return 1 - dice.mean()  # 计算Dice Loss（目标是最小化此值）


# 输入gt为(N,)
class FocalLoss(torch.nn.Module):
    def __init__(self, num_classes, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(num_classes, 1))
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.num_classes = num_classes

    def forward(self, predict, target):
        pt = F.softmax(predict, dim=1)  # softmmax获取预测概率
        class_mask = F.one_hot(target, self.num_classes)  # 获取target的one hot编码
        ids = target.view(-1, 1)
        alpha = self.alpha[ids.data.view(-1)].to(predict.device) # 注意，这里的alpha是给定的一个list(tensor
# ),里面的元素分别是每一个类的权重因子
        probs = (pt * class_mask).sum(1).view(-1, 1) # 利用onehot作为mask，提取对应的pt
        log_p = probs.log()
# 同样，原始ce上增加一个动态权重衰减因子
        loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss



