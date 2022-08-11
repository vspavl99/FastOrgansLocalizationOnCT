import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from src.data.dataset import CTDatasetPatches
from src.data.base_datasets import AMOS22Patches
from src.config import TrainConfig
from src.models.model import ModelSegmentationCT


if __name__ == '__main__':

    train_config = TrainConfig()
    dataset = AMOS22Patches(path_to_data='/home/vpavlishen/data/vpavlishen/AMOS22')

    data = CTDatasetPatches(dataset=dataset, config=train_config)

    model = smp.Unet('resnet34', encoder_weights='imagenet', classes=16, in_channels=1)
    model = ModelSegmentationCT(base_model=model, loss_function=smp.losses.DiceLoss(mode='multilabel'))

    trainer = pl.Trainer(check_val_every_n_epoch=10, max_epochs=200, accelerator="gpu", log_every_n_steps=5)
    trainer.fit(model=model, datamodule=data)
