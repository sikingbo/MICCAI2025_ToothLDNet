import click
import pytorch_lightning as pl

from pl_model import LitModel


@click.command()
@click.option('--checkpoint', type=str, default='E:/code/tooth_landmark_detection_dgcnn/runs/landmark_600_incisor_baseline/version_5/checkpoints/last.ckpt')
@click.option('--gpus', default=1)
def run(checkpoint, gpus):
    model = LitModel.load_from_checkpoint(checkpoint)
    model.hparams.args.debug = False

    trainer = pl.Trainer(gpus=gpus, max_epochs=-1)
    trainer.test(model)


if __name__ == "__main__":
    run()
