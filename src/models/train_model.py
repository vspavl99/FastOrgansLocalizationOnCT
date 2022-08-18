import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from src.config import Config
from src.models.model import ModelSegmentationCT
from src.data.dataset import CTDatasetPatches, CTDataset2D


def experiment_1():
    """
    Use 2d slices to train
    :return:
    """

    base_model = smp.Unet('resnet34', encoder_weights='imagenet', classes=16, in_channels=1)
    model = ModelSegmentationCT(base_model=base_model, loss_function=smp.losses.DiceLoss(mode='multilabel'))

    train_config = Config(batch_size=156, num_workers=6)
    datamodule_2d_images = CTDataset2D(config=train_config)

    trainer = pl.Trainer(check_val_every_n_epoch=5, max_epochs=30, accelerator="gpu", log_every_n_steps=5)
    trainer.fit(model=model, datamodule=datamodule_2d_images)


def experiment_2():
    """
    Use patches to train
    :return:
    """
    base_model = smp.Unet('resnet34', encoder_weights='imagenet', classes=16, in_channels=1)
    model = ModelSegmentationCT(base_model=base_model, loss_function=smp.losses.DiceLoss(mode='multilabel'))

    train_config = Config(batch_size=2, num_workers=6)
    datamodule_patches = CTDatasetPatches(config=train_config)

    trainer = pl.Trainer(check_val_every_n_epoch=10, max_epochs=200, accelerator="gpu", log_every_n_steps=5)
    trainer.fit(model=model, datamodule=datamodule_patches)


if __name__ == '__main__':
    experiment_2()
