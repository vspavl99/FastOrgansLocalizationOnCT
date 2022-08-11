from pathlib import Path
from typing import List


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
