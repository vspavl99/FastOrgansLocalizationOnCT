import cv2
import numpy as np
from skimage import color

from src.visualization.utils import COLOR_MAP


def visualize_mask(raw_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    image = (raw_image.transpose((1, 2, 0)) * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


    #
    #     color_mask = (np.expand_dims(mask[_class + 1], axis=2) * COLOR_MAP[_class]['color']).astype(np.uint8)
    #     image = cv2.addWeighted(image, 0.8, color_mask, 0.2, 0)

    image = (color.label2rgb(mask, image) * 255).astype(np.uint8)

    return image