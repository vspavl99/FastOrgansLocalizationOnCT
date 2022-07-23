from albumentations import Compose, Normalize, Resize
from albumentations.pytorch.transforms import ToTensorV2

from src.config import TrainConfig


class Augmentations:
    def __init__(self, config: TrainConfig):
        super().__init__()
        self.config = config

        self.height, self.width = self.config.input_size

        self.list_augmentations = self._get_augmentations()

    def __call__(self, *args, **kwargs):
        return self.list_augmentations(*args, **kwargs)

    def _get_augmentations(self) -> Compose:
        """
        Return Compose object with list of augmentations
        :return:
        """

        list_transforms = [
            Resize(always_apply=False, p=1.0, height=self.height, width=self.width),
            ToTensorV2()
        ]

        list_transforms = Compose(list_transforms)
        return list_transforms
