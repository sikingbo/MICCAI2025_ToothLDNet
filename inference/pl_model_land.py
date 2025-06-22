import torch
import trimesh
import torch.nn as nn
import pytorch_lightning as pl

from torch.utils.data import DataLoader

# from land.teethgnn import TeethLandDETR
from models.land.teethdetr import TeethDETR
from data.land.st_data import TeethLandDataset
from utils.land.loss import Criterion


def criterion(a, b):
    return torch.norm(a - b, dim=-1).mean()


class LitModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.save_hyperparameters()
        self.net = TeethDETR(args)
        self.criterion = Criterion(args)

    def forward(self, x):
        return self.net.forward(x)

    def infer(self, features1, features2, vs_offset):
        return self.net.inference(features1, features2, vs_offset)

    def training_step(self, batch, _):
        x, t_idx, g_heats, lm_labels = batch

        probs, pred_heats = self(x, t_idx, g_heats)

        loss = self.criterion(probs, lm_labels, pred_heats, g_heats)
        self.log('loss', loss, batch_size=x.size(0))
        self.log('lr', self.optimizers().param_groups[0]['lr'])

        return loss

    def validation_step(self, batch, _):
        x, t_idx, g_heats, lm_labels = batch

        probs, pred_heats = self(x, t_idx, g_heats)

        loss = self.criterion(probs, lm_labels, pred_heats, g_heats)
        self.log('val_loss', loss, True, batch_size=x.size(0))

    def test_step(self, batch, _):
        x, t_idx, g_heats, lm_labels = batch

        probs, pred_heats = self(x, t_idx, g_heats)

        loss = self.criterion(probs, lm_labels, pred_heats, g_heats)
        self.log('test_loss', loss, True, batch_size=x.size(0))

    # def configure_optimizers(self):
    #     args = self.hparams.args
    #     steps_per_epoch = (len(self.train_dataloader()) + args.gpus - 1) // args.gpus  # for multi-gpus
    #     optimizer = torch.optim.Adam(self.net.parameters(), args.lr_max, weight_decay=args.weight_decay)
    #     scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, float(args.lr_max),
    #                                                     pct_start=args.pct_start, div_factor=float(args.div_factor),
    #                                                     final_div_factor=float(args.final_div_factor),
    #                                                     epochs=args.max_epochs, steps_per_epoch=steps_per_epoch)
    #     return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
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
