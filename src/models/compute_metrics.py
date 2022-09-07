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


def compute_metrics_channels_3_images():
    test_config = Config(batch_size=138, num_workers=6, channels=3)

    model = ModelSegmentationCT.load_from_checkpoint(
        checkpoint_path="/home/vpavlishen/lightning_logs/version_54/checkpoints/epoch=79-step=12960.ckpt",
        base_model=smp.Unet('resnet34', encoder_weights='imagenet', classes=16 * test_config.channels,
                            in_channels=test_config.channels)
    )

    trainer = pl.Trainer(accelerator="gpu", log_every_n_steps=5)
    trainer.test(model=model, datamodule=CTDataset2D(config=test_config))


def compute_metrics_channels_5_images():
    test_config = Config(batch_size=98, num_workers=6, channels=5)

    model = ModelSegmentationCT.load_from_checkpoint(
        checkpoint_path="/home/vpavlishen/lightning_logs/version_56/checkpoints/epoch=9-step=2270.ckpt",
        base_model=smp.Unet('resnet34', encoder_weights='imagenet', classes=16 * test_config.channels,
                            in_channels=test_config.channels)
    )

    trainer = pl.Trainer(accelerator="cpu", log_every_n_steps=5)
    trainer.test(model=model, datamodule=CTDataset2D(config=test_config))


def compute_metrics_channels_10_images():
    test_config = Config(batch_size=68, num_workers=6, channels=10)

    model = ModelSegmentationCT.load_from_checkpoint(
        checkpoint_path="/home/vpavlishen/lightning_logs/version_58/checkpoints/epoch=19-step=6540.ckpt",
        base_model=smp.Unet('resnet34', encoder_weights='imagenet', classes=16 * test_config.channels,
                            in_channels=test_config.channels)
    )

    trainer = pl.Trainer(accelerator="cpu", log_every_n_steps=5)
    trainer.test(model=model, datamodule=CTDataset2D(config=test_config))


if __name__ == '__main__':

    compute_metrics_channels_10_images()
