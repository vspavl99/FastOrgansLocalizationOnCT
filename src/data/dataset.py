from typing import List, Tuple, Union

import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from torch.nn.functional import one_hot
from torch.utils.data import Dataset, DataLoader, random_split

from src.config import TrainConfig
from src.data.augmentation import Augmentations


class AMOS22:
    number_of_classes = 15

    def __init__(self, path_to_data: str = 'data/raw/AMOS22/imagesTr'):
        self.path_to_data = path_to_data

    def get_list_of_data(self) -> List:
        return [str(file_name) for file_name in Path(self.path_to_data).iterdir()]

    @staticmethod
    def _get_label_path(path_to_item: str) -> str:
        return path_to_item.replace('imagesTr', 'labelsTr')

    def read_data_item(self, file_name: str) -> Tuple[np.ndarray, np.ndarray]:
        image = nib.load(file_name).get_fdata()

        label_name = self._get_label_path(file_name)
        label = nib.load(label_name).get_fdata()

        return image, label


class CTORG:
    number_of_classes = 5

    def __init__(self, path_to_data: str = 'data/raw/CT-ORG'):
        self.path_to_data = path_to_data

    def get_list_of_data(self) -> List:
        return [file_name for file_name in Path(self.path_to_data).iterdir() if 'volume' in str(file_name)]

    @staticmethod
    def _get_label_path(path_to_item: Path) -> str:
        return str(path_to_item).replace('volume', 'labels')

    def read_data_item(self, file_name: Path) -> Tuple[np.ndarray, np.ndarray]:
        image = nib.load(file_name).get_fdata()

        label_name = self._get_label_path(file_name)
        label = nib.load(label_name).get_fdata()

        return image, label


class CTDataset(Dataset):
    def __init__(self, dataset: Union[CTORG, AMOS22], augmentations: Augmentations):
        self.image_names = dataset.get_list_of_data()
        self.parse_data = dataset.read_data_item

        self.number_of_classes = dataset.number_of_classes
        self.augmentations = augmentations

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_name = self.image_names[index]
        image, target = self.parse_data(image_name)

        image_preprocessed, target_preprocessed = self._preprocess_data(image=image, target=target)
        return image_preprocessed, target_preprocessed

    def _prepare_mask(self, target):
        one_hot_encoded_target = one_hot(target, num_classes=self.number_of_classes + 1)

        # Crop background class
        one_hot_encoded_target = one_hot_encoded_target[:, :, :, 1:]
        one_hot_encoded_target = one_hot_encoded_target.permute(2, 3, 0, 1)

        return one_hot_encoded_target

    def _preprocess_data(self, image: np.ndarray, target: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:

        image, target = image.transpose((0, 2, 1)).astype(np.float32), target.transpose((0, 2, 1)).astype(np.float32)
        transformed = self.augmentations(image=image, mask=target)

        transformed_image = transformed['image']
        transformed_masks = transformed['mask']

        transformed_image, transformed_masks = transformed_image.unsqueeze(1), transformed_masks.long()

        one_hot_encoded_target = self._prepare_mask(transformed_masks)

        return transformed_image[:10], one_hot_encoded_target[:10]


_dataset_dict = {
    'CTORG': CTORG,
    'AMOS22': AMOS22
}


def get_train_val_datasets(config: TrainConfig, path_to_data: str) -> Tuple[DataLoader, DataLoader]:
    """
    Create dataset, split into train and val
    :param path_to_data: path where data located
    :param config: some parameters
    :return: train and validation dataloaders
    """
    dataset_parser = _dataset_dict.get(config.dataset)(path_to_data=path_to_data)

    augmentations = Augmentations(config=config)

    dataset = CTDataset(dataset=dataset_parser, augmentations=augmentations)

    train_set_size = int(len(dataset) * config.train_set_size)
    valid_set_size = len(dataset) - train_set_size

    train_dataset, val_dataset = random_split(
        dataset=dataset,
        lengths=[train_set_size, valid_set_size],
        generator=torch.Generator().manual_seed(config.random_seed)
    )

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, num_workers=config.num_workers)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=config.batch_size, num_workers=config.num_workers)

    return train_dataloader, val_dataloader
