import cv2
import numpy as np
from skimage import color
import matplotlib.pyplot as plt

from src.visualization.utils import COLOR_MAP


def visualize_mask(raw_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    image = (raw_image.transpose((1, 2, 0)) * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    image = (color.label2rgb(mask, image) * 255).astype(np.uint8)

    return image


def plot_predictions(images, targets, predictions, num_images: int = 100):

        plt.title('Images')
        for i in range(num_images):
            plt.subplot(10, 10, i + 1)
            plt.imshow(images[i].squeeze(0), cmap='gray')
            plt.axis('off')

        plt.savefig('input.png', bbox_inches='tight')
        plt.show()

        plt.title('Targets')
        for i in range(num_images):
            plt.subplot(10, 10, i + 1)
            plt.axis('off')
            image_with_targets = visualize_mask(
                images[i].numpy(),
                np.argmax((targets[i].squeeze(0)).numpy(), axis=0)
            )
            plt.imshow(image_with_targets)

        plt.savefig('targets.png', bbox_inches='tight')
        plt.show()

        plt.title('Predictions')
        for i in range(num_images):
            plt.subplot(10, 10, i + 1)
            plt.axis('off')
            image_with_preds = visualize_mask(
                images[i].numpy(),
                np.argmax((predictions[i].squeeze(0).sigmoid() > 0.5).numpy(), axis=0)
            )
            plt.imshow(image_with_preds)

        plt.savefig('predictions.png', bbox_inches='tight')
        plt.show()
