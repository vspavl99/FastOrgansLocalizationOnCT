import pytorch_lightning as pl
from monai.networks.nets import UNETR
import segmentation_models_pytorch as smp

from src.config import Config
from src.models.model import ModelSegmentationCT
from src.data.datamodules import AMOS22DatasetPatches, CTDataset2D
from src.data.datasets import AMOS22Volumes


def compute_metrics_2d_images(model: str):

    if model == 'UNET':
        model = ModelSegmentationCT.load_from_checkpoint(
            checkpoint_path="/home/vpavlishen/lightning_logs/version_38/checkpoints/epoch=29-step=4290.ckpt",
            base_model=smp.Unet('resnet34', encoder_weights='imagenet', classes=16, in_channels=1),
        )
    elif model == 'UNETR':
        model = ModelSegmentationCT.load_from_checkpoint(
            checkpoint_path="/home/vpavlishen/lightning_logs/version_59/checkpoints/epoch=29-step=20850.ckpt",
            base_model=UNETR(in_channels=1, out_channels=16, img_size=128, spatial_dims=2)
        )
    else:
        print(f'{model} not found')
        return

    test_config = Config(batch_size=36, num_workers=1)
    trainer = pl.Trainer(accelerator="gpu", log_every_n_steps=5)
    trainer.test(
        model=model,
        datamodule=CTDataset2D(
            config=test_config, path_to_data="/home/vpavlishen/data/vpavlishen/foloct/AMOS22/processed/AMOS22"
        )
    )


def compute_metrics_patches(model: str):

    if model == 'UNET':
        model = ModelSegmentationCT.load_from_checkpoint(
            checkpoint_path="/home/vpavlishen/lightning_logs/version_7/checkpoints/epoch=89-step=8640.ckpt",
            base_model=smp.Unet('resnet34', encoder_weights='imagenet', classes=16, in_channels=1)
        )
    elif model == 'UNETR':
        model = ModelSegmentationCT.load_from_checkpoint(
            checkpoint_path="/home/vpavlishen/lightning_logs/version_61/checkpoints/epoch=79-step=14400.ckpt",
            base_model=UNETR(in_channels=1, out_channels=16, img_size=128, spatial_dims=2)
        )
    else:
        print(f'{model} not found')
        return

    test_config = Config(batch_size=2, num_workers=6)
    trainer = pl.Trainer(accelerator="gpu", log_every_n_steps=5, enable_checkpointing=False)
    trainer.test(model=model, datamodule=AMOS22DatasetPatches(config=test_config))


def compute_metrics_channels_3_images(model: str):
    test_config = Config(batch_size=138, num_workers=6, channels=3)

    if model == 'UNET':
        model = ModelSegmentationCT.load_from_checkpoint(
            checkpoint_path="/home/vpavlishen/lightning_logs/version_54/checkpoints/epoch=79-step=12960.ckpt",
            base_model=smp.Unet('resnet34', encoder_weights='imagenet',
                                classes=16 * test_config.channels, in_channels=test_config.channels)
        )
    elif model == 'UNETR':
        model = ModelSegmentationCT.load_from_checkpoint(
            checkpoint_path="/home/vpavlishen/lightning_logs/version_63/checkpoints/epoch=79-step=40480.ckpt",
            base_model=UNETR(in_channels=test_config.channels, out_channels=16 * test_config.channels,
                             img_size=128, spatial_dims=2)
        )
    else:
        print(f'{model} not found')
        return

    trainer = pl.Trainer(accelerator="gpu", log_every_n_steps=5)
    trainer.test(model=model, datamodule=CTDataset2D(config=test_config))


def compute_metrics_channels_5_images(model: str):
    test_config = Config(batch_size=98, num_workers=6, channels=5)

    if model == 'UNET':
        model = ModelSegmentationCT.load_from_checkpoint(
            checkpoint_path="/home/vpavlishen/lightning_logs/version_56/checkpoints/epoch=9-step=2270.ckpt",
            base_model=smp.Unet('resnet34', encoder_weights='imagenet', classes=16 * test_config.channels,
                                in_channels=test_config.channels)
        )
    elif model == 'UNETR':
        model = ModelSegmentationCT.load_from_checkpoint(
            checkpoint_path="/home/vpavlishen/lightning_logs/version_67/checkpoints/epoch=79-step=40480.ckpt",
            base_model=UNETR(in_channels=test_config.channels, out_channels=16 * test_config.channels,
                             img_size=128, spatial_dims=2)
        )
    else:
        print(f'{model} not found')
        return

    trainer = pl.Trainer(accelerator="gpu", log_every_n_steps=5)
    trainer.test(model=model, datamodule=CTDataset2D(config=test_config))


def compute_metrics_channels_10_images(model: str):
    test_config = Config(batch_size=3, num_workers=6, channels=10)

    if model == 'UNET':
        model = ModelSegmentationCT.load_from_checkpoint(
            checkpoint_path="/home/vpavlishen/lightning_logs/version_58/checkpoints/epoch=19-step=6540.ckpt",
            base_model=smp.Unet('resnet34', encoder_weights='imagenet', classes=16 * test_config.channels,
                                in_channels=test_config.channels)
        )
    elif model == 'UNETR':
        model = ModelSegmentationCT.load_from_checkpoint(
            checkpoint_path="/home/vpavlishen/lightning_logs/version_69/checkpoints/epoch=79-step=111200.ckpt",
            base_model=UNETR(in_channels=test_config.channels, out_channels=16 * test_config.channels,
                             img_size=128, spatial_dims=2)
        )
    else:
        print(f'{model} not found')
        return

    trainer = pl.Trainer(accelerator="gpu", log_every_n_steps=5)
    trainer.test(model=model, datamodule=CTDataset2D(config=test_config))


if __name__ == '__main__':
    compute_metrics_2d_images(model='UNET')
    compute_metrics_patches(model='UNETR')
    compute_metrics_channels_3_images(model='UNETR')
    compute_metrics_channels_10_images(model='UNETR')