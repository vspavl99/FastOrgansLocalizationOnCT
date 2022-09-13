from dataclasses import dataclass


@dataclass
class Config:

    batch_size: int = 1
    sub_batch_size: int = 10

    num_workers: int = 1
    number_of_classes: int = 16

    channels: int = 1
    patch_shape: tuple = (-1, 128, 128)
    crop_shape: tuple = (100, 128, 128)
    image_shape: tuple = (128, 128)

    model: str = ""
    checkpoint_path: str = ""

    device: str = 'cpu'
