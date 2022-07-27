from dataclasses import dataclass


@dataclass
class TrainConfig:

    batch_size: int = 10
    train_set_size: float = 0.8
    random_seed: int = 42

    input_size: tuple = (256, 256)
    num_workers: int = 2
