import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from src.data.dataset import CTDataset, AMOS22
from src.config import TrainConfig
from src.models.model import ModelSegmentationCT


if __name__ == '__main__':

    train_config = TrainConfig()
    dataset = AMOS22(path_to_data='/home/vpavlishen/data/vpavlishen/AMOS22')

    data = CTDataset(dataset=dataset, config=train_config)
    data.setup()

    print('Training:  ', len(data.train_patches_queue))
    print('Validation: ', len(data.val_patches_queue))

    model = smp.Unet('resnet18', encoder_weights='imagenet', classes=16, in_channels=1)
    model = ModelSegmentationCT(base_model=model, loss_function=smp.losses.DiceLoss(mode='multilabel'))

    trainer = pl.Trainer(check_val_every_n_epoch=5, max_epochs=50, accelerator="gpu")
    trainer.fit(model=model, datamodule=data)
