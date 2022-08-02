import numpy as np
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

from src.config import TrainConfig
from src.data.dataset import CTDataset, AMOS22
from src.models.model import ModelSegmentationCT
from src.visualization.visualize import visualize_mask

import time

if __name__ == '__main__':

    model = ModelSegmentationCT.load_from_checkpoint(
        checkpoint_path="/home/vpavlishen/lightning_logs/version_7/checkpoints/epoch=89-step=8640.ckpt",
        base_model=smp.Unet('resnet34', encoder_weights='imagenet', classes=16, in_channels=1)
    )
    model.eval()

    train_config = TrainConfig(batch_size=1, num_workers=0)

    dataset = AMOS22(path_to_data='/home/vpavlishen/data/vpavlishen/AMOS22')
    data = CTDataset(dataset=dataset, config=train_config)
    data.setup()

    val_dataloader = data.val_dataloader()
    num_images = 100

    for image, target in val_dataloader:
        print(image.shape)
        indexes = list(np.random.randint(0, len(image), num_images))
        preds = model(image).detach()

        plt.title('Images')
        for i in range(num_images):
            plt.subplot(10, 10, i + 1)
            plt.imshow(image[indexes[i]].squeeze(0), cmap='gray')
            plt.axis('off')

        plt.savefig('input.png', bbox_inches='tight')
        plt.show()

        plt.title('Targets')
        for i in range(num_images):
            plt.subplot(10, 10, i + 1)
            plt.axis('off')
            image_with_targets = visualize_mask(
                 image[indexes[i]].numpy(),
                 np.argmax((target[indexes[i]].squeeze(0)).numpy(), axis=0)
            )
            plt.imshow(image_with_targets)

        plt.savefig('targets.png', bbox_inches='tight')
        plt.show()

        plt.title('Predictions')
        for i in range(num_images):
            plt.subplot(10, 10, i + 1)
            plt.axis('off')
            image_with_preds = visualize_mask(
                image[indexes[i]].numpy(),
                np.argmax((preds[indexes[i]].squeeze(0).sigmoid() > 0.5).numpy(), axis=0)
            )
            plt.imshow(image_with_preds)

        plt.savefig('predictions.png', bbox_inches='tight')
        plt.show()



        time.sleep(10)