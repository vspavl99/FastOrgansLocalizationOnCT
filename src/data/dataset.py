from typing import List, Tuple, Union

import torch
import numpy as np
import nibabel as nib
import torchio as tio
from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from src.config import TrainConfig
from src.data.augmentation import Augmentations


class AMOS22:
    number_of_classes = 15

    def __init__(self, path_to_data: str = 'data/raw/AMOS22/'):
        self.path_to_data = Path(path_to_data)

    @staticmethod
    def _filter_files(filename):
        return not str(filename.name).startswith('.')

    def _get_list_of_data(self, folder: Path) -> list:
        path_to_files = self.path_to_data / folder
        return list(filter(self._filter_files, path_to_files.iterdir()))

    def get_images(self) -> list:
        return self._get_list_of_data(Path('imagesTr'))

    def get_labels(self) -> list:
        return self._get_list_of_data(Path('labelsTr'))


class CTORG:
    number_of_classes = 5

    def __init__(self, path_to_data: str = 'data/raw/CT-ORG'):
        self.path_to_data = Path(path_to_data)

    def get_list_of_data(self) -> List:
        return [file_name for file_name in Path(self.path_to_data).iterdir()
                if 'volume' in str(file_name) and not str(file_name.name).startswith('.')]

    @staticmethod
    def _get_label_path(path_to_item: Path) -> str:
        return str(path_to_item).replace('volume', 'labels')

    def read_data_item(self, file_name: Path) -> Tuple[np.ndarray, np.ndarray]:
        image = nib.load(file_name).get_fdata()

        label_name = self._get_label_path(file_name)
        label = nib.load(label_name).get_fdata()

        return image, label


class CTDataset(pl.LightningDataModule):
    def __init__(self, dataset: Union[CTORG, AMOS22], config: TrainConfig):
        super().__init__()

        self.config = config

        self.dataset = dataset
        self.augmentations = Augmentations(config=config)

        self.subjects = None
        self.val_patches_queue = None
        self.train_patches_queue = None

        self.shape = 256
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
            tio.EnsureShapeMultiple(8),  # for the U-Net
            tio.OneHot(num_classes=self.number_of_classes + 1),
        ])
        return preprocess

    @staticmethod
    def get_augmentation_transform():
        augment = tio.Compose([
            tio.RandomAffine(),
            tio.RandomGamma(p=0.5),
            tio.RandomNoise(p=0.5),
            tio.RandomMotion(p=0.1),
            tio.RandomBiasField(p=0.25),
        ])
        return augment

    def setup(self, stage=None):
        self.prepare_data()

        num_subjects = len(self.subjects)
        num_train_subjects = int(round(num_subjects * self.config.train_set_size))
        num_val_subjects = num_subjects - num_train_subjects

        train_subjects, val_subjects = random_split(
            dataset=self.subjects,
            lengths=[num_train_subjects, num_val_subjects],
            generator=torch.Generator().manual_seed(self.config.random_seed)
        )

        preprocess = self._preprocess_data()
        augment = self.get_augmentation_transform()
        transform = tio.Compose([preprocess, augment])

        sampler = tio.data.UniformSampler((1, 256, 256))

        train_subjects_dataset = tio.SubjectsDataset(train_subjects, transform=transform)
        self.train_patches_queue = tio.Queue(
            subjects_dataset=train_subjects_dataset,
            max_length=1,
            samples_per_volume=1,
            sampler=sampler,
            num_workers=4,
        )

        val_subjects_dataset = tio.SubjectsDataset(val_subjects, transform=preprocess)
        self.val_patches_queue = tio.Queue(
            subjects_dataset=val_subjects_dataset,
            max_length=4,
            samples_per_volume=2,
            sampler=sampler,
            num_workers=2,
        )

    def train_dataloader(self):
        return DataLoader(self.train_patches_queue, self.config.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_patches_queue, self.config.batch_size)
