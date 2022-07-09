from dataclasses import dataclass


@dataclass
class TrainConfig:
    dataset: str

    batch_size: int = 1
    train_set_size: float = 0.8
    random_seed: int = 42

