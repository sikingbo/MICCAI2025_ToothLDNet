import torch
import trimesh
import torch.nn as nn
import pytorch_lightning as pl

from torch.utils.data import DataLoader

from models.teethdetr import TeethDETR
from data.st_data import TeethLandDataset
from utils.loss import Criterion


def criterion(a, b):
    return torch.norm(a - b, dim=-1).mean()


class LitModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.save_hyperparameters()
        self.net = TeethDETR(args)
        self.criterion = Criterion(args)


    def forward(self, features1, features2, mapping):
        return self.net.forward(features1, features2, mapping)

    def infer(self, features1, features2, mapping):
        p_xyz, p_labels = self.net.inference(features1, features2, mapping)
        return p_xyz, p_labels

    def training_step(self, batch, _):
        features1, features2, g_heats, lm_labels, mapping, gt_offsets, vs_mean = batch

        probs, pred_heats, offsets = self(features1, features2, vs_mean)

        l2 = torch.norm(offsets - gt_offsets, dim=1).mean()
        loss_detr = self.criterion(probs, lm_labels, pred_heats, g_heats)
        loss = loss_detr + l2 * 0.1
        self.log('loss/l2', l2, True, batch_size=features1.size(0))
        self.log('loss/detr', loss_detr, True, batch_size=features1.size(0))
        self.log('loss', loss, batch_size=features1.size(0))
        self.log('lr', self.optimizers().param_groups[0]['lr'])
        return loss

    def validation_step(self, batch, _):
        features1, features2, g_heats, lm_labels, mapping, gt_offsets, vs_mean = batch

        probs, pred_heats, offsets = self(features1, features2, vs_mean)

        l2 = torch.norm(offsets - gt_offsets, dim=1).mean()
        loss_detr = self.criterion(probs, lm_labels, pred_heats, g_heats)
        loss = loss_detr + l2 * 0.1
        self.log('val_loss', loss, True, batch_size=features1.size(0))

    def test_step(self, batch, _):
        features1, features2, g_heats, lm_labels, mapping, gt_offsets, vs_mean = batch

        probs, pred_heats, offsets = self(features1, features2, vs_mean)

        l2 = torch.norm(offsets - gt_offsets, dim=1).mean()
        loss_detr = self.criterion(probs, lm_labels, pred_heats, g_heats)
        loss = loss_detr + l2 * 0.1
        self.log('test_loss', loss, True, batch_size=features1.size(0))

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
        return DataLoader(TeethLandDataset(args, args.train_file, True),
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=args.train_workers,
                          pin_memory=True)

    def val_dataloader(self):
        args = self.hparams.args
        return DataLoader(TeethLandDataset(args, args.val_file, False),
                          batch_size=args.batch_size,
                          shuffle=False,
                          num_workers=args.val_workers,
                          pin_memory=True)

    def test_dataloader(self):
        args = self.hparams.args
        return DataLoader(TeethLandDataset(args, args.test_file, False),
                          batch_size=args.batch_size,
                          shuffle=False,
                          num_workers=args.test_workers,
                          pin_memory=True)


class LitModelInference(LitModel):
    def forward(self, x):
        return torch.argmax(self.net(x), dim=2)
