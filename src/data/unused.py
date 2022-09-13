from typing import Union, List
from pathlib import Path

import torch
import torchio as tio
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from src.config import Config
from src.data.datasets import AMOS22, AMOS22Patches


class CTORG:
    number_of_classes = 5

    def __init__(self, path_to_data: str = 'data/raw/CT-ORG'):
        self.path_to_data = Path(path_to_data)

    def get_list_of_data(self) -> list:
        return [file_name for file_name in Path(self.path_to_data).iterdir()
                if 'volume' in str(file_name) and not str(file_name.name).startswith('.')]

    @staticmethod
    def _get_label_path(path_to_item: Path) -> str:
        return str(path_to_item).replace('volume', 'labels')


class CTDatasetPatches_old(pl.LightningDataModule):
    def __init__(self, dataset: Union[CTORG, AMOS22], config: Config):
        super().__init__()

        self.config = config

        self.dataset = dataset
        # self.augmentations = Augmentations(config=config)

        self.subjects = None
        self.val_subjects_dataset = None
        self.train_subjects_dataset = None

        self.shape = config.patch_shape
        self.number_of_classes = dataset.number_of_classes

    def prepare_data(self):
        image_training_paths, label_training_paths = self.dataset.get_images(), self.dataset.get_labels()

        self.subjects = []
        for image_path, label_path in zip(image_training_paths, label_training_paths):
            subject = tio.Subject(
                image=tio.ScalarImage(image_path),
                label=tio.LabelMap(label_path)
            )
            self.subjects.append(subject)

    def _preprocess_data(self):
        preprocess = tio.Compose([
            tio.RescaleIntensity((-1, 1)),
            tio.CropOrPad(self.shape),
            tio.OneHot(num_classes=self.number_of_classes + 1),
        ])
        return preprocess

    @staticmethod
    def get_augmentation_transform():
        augment = tio.Compose([
            # tio.RandomAffine(),
            # tio.RandomGamma(p=0.5),
            tio.RandomNoise(p=0.5)
            # tio.RandomMotion(p=0.1),
            # tio.RandomBiasField(p=0.25),
        ])
        return augment

    def _split_train_val_data(self):
        num_subjects = len(self.subjects)

        num_train_subjects = int(round(num_subjects * self.config.train_set_size))
        num_val_subjects = num_subjects - num_train_subjects

        train_subjects, val_subjects = random_split(
            dataset=self.subjects,
            lengths=[num_train_subjects, num_val_subjects],
            generator=torch.Generator().manual_seed(self.config.random_seed)
        )
        return train_subjects, val_subjects

    def setup(self, stage=None):
        self.prepare_data()

        train_subjects, val_subjects = self._split_train_val_data()

        preprocess = self._preprocess_data()
        augment = self.get_augmentation_transform()
        transform = tio.Compose([preprocess, augment])

        self.train_subjects_dataset = tio.SubjectsDataset(train_subjects, transform=transform)
        self.val_subjects_dataset = tio.SubjectsDataset(val_subjects, transform=preprocess)

    def train_dataloader(self):
        return DataLoader(dataset=self.train_subjects_dataset,
                          batch_size=self.config.batch_size,
                          num_workers=self.config.num_workers,
                          collate_fn=self._collate_fn)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_subjects_dataset,
                          batch_size=self.config.batch_size,
                          num_workers=self.config.num_workers,
                          collate_fn=self._collate_fn)

    @staticmethod
    def _collate_fn(data):
        batch_image = torch.cat([image['image']['data'] for image in data], dim=3).permute(3, 0, 1, 2)
        batch_target = torch.cat([target['label']['data'] for target in data], dim=3).permute(3, 0, 1, 2)

        return batch_image, batch_target



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