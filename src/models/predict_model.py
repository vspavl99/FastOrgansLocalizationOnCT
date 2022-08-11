import numpy as np
import segmentation_models_pytorch as smp

from src.config import TrainConfig
from src.data.dataset import CTDatasetPatches
from src.data.base_datasets import AMOS22Patches
from src.models.model import ModelSegmentationCT
from src.visualization.visualize import plot_predictions

SUBSET_SIZE = 100

if __name__ == '__main__':
    num_images = SUBSET_SIZE

    model = ModelSegmentationCT.load_from_checkpoint(
        checkpoint_path="/home/vpavlishen/lightning_logs/version_7/checkpoints/epoch=89-step=8640.ckpt",
        base_model=smp.Unet('resnet34', encoder_weights='imagenet', classes=16, in_channels=1)
    )
    model.eval()
    model.to('cuda')

    train_config = TrainConfig(batch_size=2, num_workers=0)
    dataset = AMOS22Patches(path_to_data='/home/vpavlishen/data/vpavlishen/AMOS22')
    data = CTDatasetPatches(dataset=dataset, config=train_config)
    data.setup()

    for image, target in data.val_dataloader():
        print(image.shape)
        indexes = list(np.random.randint(0, len(image), num_images))
        preds = model(image.to('cuda')).detach().cpu()

        images_subset, targets_subset, preds_subset = image[indexes], target[indexes], preds[indexes]

        plot_predictions(images=images_subset, targets=targets_subset, predictions=preds_subset, num_images=num_images)
