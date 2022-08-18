import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from src.config import Config
from src.models.model import ModelSegmentationCT
from src.data.dataset import CTDatasetPatches, CTDataset2D


def compute_metrics_2d_images():

    model = ModelSegmentationCT.load_from_checkpoint(
        checkpoint_path="/home/vpavlishen/lightning_logs/version_38/checkpoints/epoch=29-step=4290.ckpt",
        base_model=smp.Unet('resnet34', encoder_weights='imagenet', classes=16, in_channels=1)
    )

    test_config = Config(batch_size=156, num_workers=6)
    trainer = pl.Trainer(accelerator="gpu", log_every_n_steps=5)
    trainer.test(model=model, datamodule=CTDataset2D(config=test_config))


def compute_metrics_patches():

    model = ModelSegmentationCT.load_from_checkpoint(
        checkpoint_path="/home/vpavlishen/lightning_logs/version_7/checkpoints/epoch=89-step=8640.ckpt",
        base_model=smp.Unet('resnet34', encoder_weights='imagenet', classes=16, in_channels=1)
    )

    test_config = Config(batch_size=2, num_workers=6)
    trainer = pl.Trainer(accelerator="gpu", log_every_n_steps=5)
    trainer.test(model=model, datamodule=CTDatasetPatches(config=test_config))


if __name__ == '__main__':

    compute_metrics_patches()
