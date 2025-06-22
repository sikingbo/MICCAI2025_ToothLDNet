import torch
import trimesh
import torch.nn as nn
import pytorch_lightning as pl

from torch.utils.data import DataLoader
import torch.nn.functional as F

from models.gnn.teethgnn import TeethGNN
from data.gnn.st_data import Teeth3DS


def find_peak(heatmap, xs):
    return torch.stack([xs[idx][:, max_idx].T for idx, max_idx in enumerate(torch.argmax(heatmap, axis=2))])


def criterion(a, b):
    return torch.norm(a - b, dim=-1).mean()


class LitModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.save_hyperparameters()
        self.net = TeethGNN(args)
        # self.train_iou = pl.metrics.IoU(17, ignore_index=0)
        # self.val_iou = pl.metrics.IoU(17, ignore_index=0)
        # self.test_iou = pl.metrics.IoU(17, ignore_index=0)

    def forward(self, x):
        return self.net(x)

    def infer(self, x):
        return self.net.inference(x)

    def training_step(self, batch, _):
        delta = self.hparams.args.delta
        x, y, y_offsets = batch
        out, t, out_offsets = self(x)

        ce = F.cross_entropy(out, y)
        l2 = torch.norm(out_offsets - y_offsets, dim=1).mean()
        loss = ce + delta * l2
        self.log('loss/ce', ce, True)
        self.log('loss/l2', l2, True)
        self.log('loss', loss)
        self.log('lr', self.optimizers().param_groups[0]['lr'])
        return loss

    def validation_step(self, batch, _):
        delta = self.hparams.args.delta
        x, y, y_offsets = batch
        out, t, out_offsets = self(x)

        ce = F.cross_entropy(out, y)
        l2 = torch.norm(out_offsets - y_offsets, dim=1).mean()
        loss = ce + delta * l2
        self.log('val_loss/ce', ce, True)
        self.log('val_loss/l2', l2, True)
        self.log('val_loss', loss, True)
        # self.val_iou(F.softmax(out, 1), y)
        # self.log('val_iou', self.val_iou, on_step=True, on_epoch=True, prog_bar=True)

    def test_step(self, batch, _):
        delta = self.hparams.args.delta
        x, y, y_offsets = batch
        out, t, out_offsets = self(x)

        ce = F.cross_entropy(out, y)
        l2 = torch.norm(out_offsets - y_offsets, dim=1).mean()
        loss = ce + delta * l2
        self.log('test_loss/ce', ce, True)
        self.log('test_loss/l2', l2, True)
        self.log('test_loss', loss, True)
        # self.test_iou(F.softmax(out, 1), y)
        # self.log('test_iou', self.test_iou, on_step=True, on_epoch=True, prog_bar=True)

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
