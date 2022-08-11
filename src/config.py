from dataclasses import dataclass


@dataclass
class TrainConfig:

    batch_size: int = 2
    patch_shape: tuple = (128, 128, 100)
    num_workers: int = 6

    train_set_size: float = 0.8
    random_seed: int = 42

    image_shape: tuple = (128, 128)
