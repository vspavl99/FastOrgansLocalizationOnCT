import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from src.config import Config
from src.models.model import ModelSegmentationCT
from src.data.datamodules import AMOS22DatasetPatches, CTDataset2D, AMOS22DatasetVolumes

from monai.networks.nets import UNETR


def experiment_1():
    """
    Use 2d slices to train
    :return:
    """

    base_model = smp.Unet('resnet34', encoder_weights='imagenet', classes=16, in_channels=1)
    # base_model = UNETR(in_channels=1, out_channels=16, img_size=256, spatial_dims=2)

    model = ModelSegmentationCT(base_model=base_model, loss_function=smp.losses.DiceLoss(mode='multilabel'))
    train_config = Config(batch_size=120, num_workers=6, image_shape=(256, 256))
    datamodule_2d_images = CTDataset2D(
        config=train_config,
        path_to_data='/home/vpavlishen/data/vpavlishen/foloct/AMOS22/processed/AMOS22'
    )

    trainer = pl.Trainer(check_val_every_n_epoch=5, max_epochs=150, accelerator="gpu", log_every_n_steps=5)
    trainer.fit(model=model, datamodule=datamodule_2d_images)


def experiment_2():
    """
    Use patches to train
    :return:
    """
    train_config = Config(batch_size=1, num_workers=6, patch_shape=(-1, 128, 128), crop_shape=(100, 128, 128))

    # base_model = smp.Unet('resnet34', encoder_weights='imagenet', classes=16, in_channels=1)
    base_model = UNETR(in_channels=1, out_channels=16, img_size=128, spatial_dims=2)

    model = ModelSegmentationCT(base_model=base_model, loss_function=smp.losses.DiceLoss(mode='multilabel'))

    datamodule_patches = AMOS22DatasetVolumes(path_to_data='/home/vpavlishen/data/vpavlishen/foloct/AMOS22/raw/AMOS22',
                                              config=train_config)

    trainer = pl.Trainer(check_val_every_n_epoch=10, max_epochs=150, accelerator="gpu", log_every_n_steps=5)
    trainer.fit(model=model, datamodule=datamodule_patches)


def experiment_3():
    """
    Use channels to train
    :return:
    """
    train_config = Config(batch_size=16, num_workers=6, channels=10)

    # base_model = smp.Unet('resnet34', encoder_weights='imagenet', classes=16 * train_config.channels,
    #                       in_channels=train_config.channels)
    base_model = UNETR(in_channels=train_config.channels, out_channels=16 * train_config.channels,
                       img_size=128, spatial_dims=2)

    model = ModelSegmentationCT(base_model=base_model, loss_function=smp.losses.DiceLoss(mode='multilabel'))

    datamodule = CTDataset2D(config=train_config)

    trainer = pl.Trainer(check_val_every_n_epoch=10, max_epochs=80, accelerator="gpu", log_every_n_steps=5)
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == '__main__':
    experiment_2()
