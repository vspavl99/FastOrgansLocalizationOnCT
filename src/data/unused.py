from typing import Union

import torch
import torchio as tio
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from src.config import Config
from src.data.base_datasets import AMOS22Patches, CTORG


class CTDatasetPatches_old(pl.LightningDataModule):
    def __init__(self, dataset: Union[CTORG, AMOS22Patches], config: Config):
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
