import glob
from typing import List, Tuple
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.functional import one_hot


class AMOS22Patches:
    number_of_classes = 16  # 15 classes + background
    file_type = '*.nii.gz'

    def __init__(self, path_to_data: str = ''):
        self.path_to_data = Path(path_to_data)

    def _get_list_of_data(self, folder: Path) -> list:
        path_to_files = self.path_to_data / folder
        template = str(Path(path_to_files).joinpath(self.file_type))
        return sorted(glob.glob(template))

    def get_images(self, stage: str = None) -> list:
        return self._get_list_of_data(Path(stage).joinpath('imagesTr'))

    def get_labels(self, stage: str = None) -> list:
        return self._get_list_of_data(Path(stage).joinpath('labelsTr'))

    def get_data_paths(self, stage: str = None) -> list:
        image_training_paths, label_training_paths = self.get_images(stage=stage), self.get_labels(stage=stage)
        data_paths = list(zip(image_training_paths, label_training_paths))
        return data_paths


class AMOS22Images2D(Dataset, AMOS22Patches):
    file_type = '*/*.npz'

    def __init__(self, path_to_data: str = '', stage: str = None, transforms=None, channels=1):
        super().__init__()
        self.channels = channels
        self.path_to_data = Path(path_to_data)
        self.transforms = transforms
        self.data_paths = self.get_data_paths(stage)

    def __len__(self) -> int:
        return len(self.data_paths)

    def _get_indexes(self, index) -> np.array:
        length = self.__len__()

        start = index - self.channels // 2
        finish = index + self.channels // 2 - (1 - self.channels % 2)

        if finish >= length:
            indexes = np.concatenate(
                (np.arange(start, length), np.arange(finish - length + 1))
            )
        else:
            indexes = np.arange(start, finish + 1)

        return indexes

    def _get_prepared_item(self, index) -> tuple:
        image_path, label_path = self.data_paths[index]
        image, label = np.load(image_path)['arr_0'].astype(np.float32), np.load(label_path)['arr_0'].astype(np.uint8)

        transformed = self.transforms(image=image, mask=label)
        transformed_image, transformed_masks = transformed['image'], transformed['mask'].long()

        transformed_target = self._decode_label(transformed_masks).permute(2, 0, 1)

        return transformed_image, transformed_target

    def __getitem__(self, index) -> tuple:
        indexes = self._get_indexes(index)
        transformed_images, transformed_labels = [], []

        if len(indexes) == 1:
            return self._get_prepared_item(index)

        for _index in indexes:
            try:
                transformed_image, transformed_target = self._get_prepared_item(_index)
            except Exception as e:
                print(e, indexes)

            transformed_images.append(transformed_image)
            transformed_labels.append(transformed_target)

        transformed_images = np.concatenate(transformed_images, axis=0)
        transformed_labels = np.stack(transformed_labels, axis=1)

        _, _, width, height = transformed_labels.shape
        transformed_labels = transformed_labels.reshape((-1, width, height))

        return transformed_images, transformed_labels

    def _decode_label(self, label: torch.Tensor) -> torch.Tensor:
        target = one_hot(label, num_classes=self.number_of_classes)
        return target


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
