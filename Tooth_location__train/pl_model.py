import torch
import numpy as np
import trimesh
import torch.nn as nn
import pytorch_lightning as pl

from torch.utils.data import DataLoader
import torch.nn.functional as F

from models.teethgnn import TeethGNN
from data.st_data import Teeth3DS
from scripts.segment import segment_patch
from utils.TeethGNN.cluster import Cluster
from utils.DETR.loss import Criterion


def find_peak(heatmap, xs):
    return torch.stack([xs[idx][:, max_idx].T for idx, max_idx in enumerate(torch.argmax(heatmap, axis=2))])


def criterion(a, b):
    return torch.norm(a - b, dim=-1).mean()


class LitModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.save_hyperparameters()
        self.net = TeethGNN(args)
        self.cluster = Cluster()

        self.oh_loss = nn.MSELoss()
        self.nch_loss = nn.CrossEntropyLoss()
        self.ch_loss = Criterion(args)
        self.query_num = args.query_num
        # self.train_iou = pl.metrics.IoU(17, ignore_index=0)
        # self.val_iou = pl.metrics.IoU(17, ignore_index=0)
        # self.test_iou = pl.metrics.IoU(17, ignore_index=0)

    def forward(self, x):
        return self.net(x)

    def infer(self, x):
        p_labels = self.net.inference(x)
        return p_labels

    def training_step(self, batch, _):
        delta = self.hparams.args.delta
        x, labels, gt_offsets = batch
        out, probs, offsets = self(x)

        labels = labels.view(labels.size(0), labels.size(1), -1)  # B,K,N*2
        l2 = torch.norm(offsets - gt_offsets, dim=1).mean()
        ce = self.ch_loss(labels, probs, out)
        loss = ce + l2 * delta
        self.log('loss/ce', ce, True)
        self.log('loss/l2', l2, True)
        self.log('loss', loss)
        self.log('lr', self.optimizers().param_groups[0]['lr'])
        return loss

    def validation_step(self, batch, _):
        delta = self.hparams.args.delta
        x, labels, gt_offsets = batch
        out, probs, offsets = self(x)

        labels = labels.view(labels.size(0), labels.size(1), -1)  # B,K,N*2
        l2 = torch.norm(offsets - gt_offsets, dim=1).mean()
        ce = self.ch_loss(labels, probs, out)
        loss = ce + l2 * delta
        self.log('val_loss/ce', ce, True)
        # self.log('val_loss/l2', l2, True)
        self.log('val_loss', loss)

    def test_step(self, batch, _):
        delta = self.hparams.args.delta
        x, labels, gt_offsets = batch
        out, probs, offsets = self(x)

        labels = labels.view(labels.size(0), labels.size(1), -1)  # B,K,N*2
        l2 = torch.norm(offsets - gt_offsets, dim=1).mean()
        ce = self.ch_loss(labels, probs, out)
        loss = ce + l2 * delta
        self.log('test_loss/ce', ce, True)
        # self.log('test_loss/l2', l2, True)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        args = self.hparams.args
        steps_per_epoch = (len(self.train_dataloader()) + args.gpus - 1) // args.gpus  # for multi-gpus
        optimizer = torch.optim.Adam(self.net.parameters(), args.lr_max, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, float(args.lr_max),
                                                        pct_start=args.pct_start, div_factor=float(args.div_factor),
                                                        final_div_factor=float(args.final_div_factor),
                                                        epochs=args.max_epochs, steps_per_epoch=steps_per_epoch)
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    def configure_optimizers(self):
        args = self.hparams.args
        optimizer = torch.optim.Adam(self.net.parameters(), args.lr_max, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, float(args.lr_max),
                                                        pct_start=args.pct_start, div_factor=float(args.div_factor),
                                                        final_div_factor=float(args.final_div_factor),
                                                        epochs=args.max_epochs,
                                                        steps_per_epoch=len(self.train_dataloader()))
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    def train_dataloader(self):
        args = self.hparams.args
        return DataLoader(Teeth3DS(args, args.train_file, True),
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=args.train_workers,
                          pin_memory=True)

    def val_dataloader(self):
        args = self.hparams.args
        return DataLoader(Teeth3DS(args, args.val_file, False),
                          batch_size=args.batch_size,
                          shuffle=False,
                          num_workers=args.val_workers,
                          pin_memory=True)

    def test_dataloader(self):
        args = self.hparams.args
        return DataLoader(Teeth3DS(args, args.test_file, False),
                          batch_size=args.batch_size,
                          shuffle=False,
                          num_workers=args.test_workers,
                          pin_memory=True)
