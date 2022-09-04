from dataclasses import dataclass


@dataclass
class Config:

    batch_size: int = 1
    num_workers: int = 1

    channels: int = 1
    patch_shape: tuple = (128, 128, 100)
    image_shape: tuple = (128, 128)

