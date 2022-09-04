from typing import List

import torch
import torchio as tio
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.config import Config
from src.data.base_datasets import AMOS22Patches, AMOS22Images2D
from src.data.augmentation import Augmentations


class CTDatasetPatches(pl.LightningDataModule):
    def __init__(self, config: Config, path_to_data: str = 'data/vpavlishen/raw/AMOS22'):
        super().__init__()

        self.config = config
        self.path_to_data = path_to_data
        self.data_parser = AMOS22Patches(path_to_data=self.path_to_data)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self._patch_shape = config.patch_shape

    def _preprocess_data(self):
        preprocess = tio.Compose([
            tio.RescaleIntensity((-1, 1)),
            tio.CropOrPad(self._patch_shape),
            tio.OneHot(num_classes=self.data_parser.number_of_classes),
        ])
        return preprocess

    @staticmethod
    def _get_augmentation_transform():
        augmentations = tio.Compose([
            # tio.RandomAffine(),
            # tio.RandomGamma(p=0.5),
            tio.RandomNoise(p=0.5)
            # tio.RandomMotion(p=0.1),
            # tio.RandomBiasField(p=0.25),
        ])
        return augmentations

    @staticmethod
    def _make_tio_subjects(data: list) -> List[tio.Subject]:
        subjects = []
        for image_path, label_path in data:
            subject = tio.Subject(
                image=tio.ScalarImage(image_path),
                label=tio.LabelMap(label_path)
            )
            subjects.append(subject)
        return subjects

    def _get_transforms(self, stage: str = None):
        preprocess = self._preprocess_data()
        augmentations = self._get_augmentation_transform()
        transform = tio.Compose([preprocess, augmentations])
        return transform

    def setup(self, stage: str = None):

        transform = self._get_transforms(stage=stage)

        if stage == 'fit' or stage is None:
            train_data_paths = self.data_parser.get_data_paths(stage='train')
            train_subjects = self._make_tio_subjects(train_data_paths)
            self.train_dataset = tio.SubjectsDataset(train_subjects, transform=transform)

            val_data_paths = self.data_parser.get_data_paths(stage='val')
            val_subjects = self._make_tio_subjects(val_data_paths)
            self.val_dataset = tio.SubjectsDataset(val_subjects, transform=transform)

        if stage == 'test' or stage is None:
            test_data_paths = self.data_parser.get_data_paths(stage='test')
            test_subjects = self._make_tio_subjects(test_data_paths)
            self.test_dataset = tio.SubjectsDataset(test_subjects, transform=transform)

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset,
                          batch_size=self.config.batch_size,
                          num_workers=self.config.num_workers,
                          collate_fn=self._collate_fn)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset,
                          batch_size=self.config.batch_size,
                          num_workers=self.config.num_workers,
                          collate_fn=self._collate_fn)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset,
                          batch_size=self.config.batch_size,
                          num_workers=self.config.num_workers,
                          collate_fn=self._collate_fn)

    @staticmethod
    def _collate_fn(data):
        batch_image = torch.cat([image['image']['data'] for image in data], dim=3).permute(3, 0, 1, 2)
        batch_target = torch.cat([target['label']['data'] for target in data], dim=3).permute(3, 0, 1, 2)

        return batch_image, batch_target


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
                          num_workers=self.config.num_workers)
