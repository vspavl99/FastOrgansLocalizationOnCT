import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from src.data.dataset import get_train_val_datasets
from src.config import TrainConfig
from src.models.model import ModelSegmentationCT


if __name__ == '__main__':

    train_config = TrainConfig(dataset='AMOS22')
    train_dataloader, val_dataloader = get_train_val_datasets(train_config)

    model = smp.Unet('resnet34', encoder_weights='imagenet', classes=15)
    model = ModelSegmentationCT(base_model=model, loss_function=torch.nn.BCEWithLogitsLoss())

    trainer = pl.Trainer(check_val_every_n_epoch=5, max_epochs=10)
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
