import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from .matcher import HungarianMatcher


class Criterion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.matcher = HungarianMatcher(cost_prob=100, cost_heat=1)
        self.loss_h = nn.MSELoss()#nn.L1Loss()
        # self.loss_c = nn.CrossEntropyLoss()
        self.loss_c = FocalLoss(num_classes=7)

    def forward(self, probs, label, heats, g_heat):
        indices = self.matcher(probs, label, heats, g_heat)

        loss = 0
        for i, idx in enumerate(indices):
            gheat = g_heat[i, idx[1], :]
            pheat = heats[i, idx[0], :]
            loss_h = self.loss_h(pheat, gheat)

            prob = probs[i]
            gtc = label[i, idx[1]]

            # back as zero
            # gclass = torch.zeros(prob.shape[0], dtype=torch.long).to(prob.device)
            # gclass[idx[0]] = gtc

            # back as num_class
            gclass = - torch.ones(prob.shape[0], dtype=torch.long).to(prob.device)
            gclass[idx[0]] = gtc
            gclass = torch.where(gclass == -1, 0, gclass)

            # cross entropy loss
            # one_hot = F.one_hot(gclass, num_classes=self.num_classes+1).float()
            # loss_c = self.loss_c(prob, one_hot)
            # focal loss
            loss_c = self.loss_c(prob, gclass)

            loss = loss + loss_h * 30 + loss_c

        return loss


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
        pt = F.softmax(predict, dim=1) # softmmax获取预测概率
        class_mask = F.one_hot(target, self.num_classes) #获取target的one hot编码
        ids = target.view(-1, 1)
        alpha = self.alpha[ids.data.view(-1)].to(predict.device) # 注意，这里的alpha是给定的一个list(tensor
#),里面的元素分别是每一个类的权重因子
        probs = (pt * class_mask).sum(1).view(-1, 1) # 利用onehot作为mask，提取对应的pt
        log_p = probs.log()
# 同样，原始ce上增加一个动态权重衰减因子
        loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss