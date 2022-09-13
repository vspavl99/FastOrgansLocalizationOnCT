import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.config import Config
from src.data.datasets import AMOS22Images2D, AMOS22Volumes
from src.data.augmentation import Augmentations


class CTDataset2D(pl.LightningDataModule):
    def __init__(self, config: Config, path_to_data: str = 'data/vpavlishen/processed/AMOS22'):
        super().__init__()

        self.config = config
        self.path_to_data = path_to_data

        self.augmentations = Augmentations(config=config)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str = None):

        if stage == 'fit' or stage is None:
            self.train_dataset = AMOS22Images2D(path_to_data=self.path_to_data, stage='train',
                                                transforms=self.augmentations, channels=self.config.channels)
            self.val_dataset = AMOS22Images2D(path_to_data=self.path_to_data, stage='val',
                                              transforms=self.augmentations, channels=self.config.channels)

        if stage == 'test' or stage is None:
            self.test_dataset = AMOS22Images2D(path_to_data=self.path_to_data, stage='test',
                                               transforms=self.augmentations, channels=self.config.channels)

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset,
                          batch_size=self.config.batch_size,
                          num_workers=self.config.num_workers,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset,
                          batch_size=self.config.batch_size,
                          num_workers=self.config.num_workers)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset,
                          batch_size=self.config.batch_size,
                          num_workers=self.config.num_workers, shuffle=True)


class AMOS22DatasetVolumes(pl.LightningDataModule):
    """
    Load whole volume from patient
    """

    def __init__(self, config: Config, path_to_data: str = 'data/vpavlishen/processed/AMOS22'):
        super().__init__()

        self.config = config
        self.path_to_data = path_to_data

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str = None):

        if stage == 'fit' or stage is None:
            self.train_dataset = AMOS22Volumes(path_to_data=self.path_to_data, stage='train', config=self.config)
            self.val_dataset = AMOS22Volumes(path_to_data=self.path_to_data, stage='val', config=self.config)

        if stage == 'test' or stage is None:
            self.test_dataset = AMOS22Volumes(path_to_data=self.path_to_data, stage='test', config=self.config)

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset,
                          batch_size=None,
                          num_workers=self.config.num_workers,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset,
                          batch_size=None,
                          num_workers=self.config.num_workers)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset,
                          batch_size=None,
                          num_workers=self.config.num_workers, shuffle=True)

    @staticmethod
    def _collate_fn(data):
        batch_image = torch.cat([image['image']['data'] for image in data], dim=3).permute(3, 0, 1, 2)
        batch_target = torch.cat([target['label']['data'] for target in data], dim=3).permute(3, 0, 1, 2)

        return batch_image, batch_target