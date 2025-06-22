import click
import pytorch_lightning as pl

from pl_model import LitModel


@click.command()
@click.option('--checkpoint', type=str, default='D:\code/teeth\TeethLandCallenge\TeethGNN/runs/all_tooth/version_0/checkpoints/last.ckpt')
@click.option('--gpus', default=1)
def run(checkpoint, gpus):
    model = LitModel.load_from_checkpoint(checkpoint)
    model.hparams.args.debug = False

    trainer = pl.Trainer(gpus=gpus, max_epochs=-1)
    trainer.test(model)


if __name__ == "__main__":
    run()
