import gzip
import shutil
from pathlib import Path
from typing import NoReturn


def unzip_files(source_folder: str, destination_folder: str) -> NoReturn:
    """
    Unzips files from source_folder and save them to destination_folder
    :param destination_folder:
    :param source_folder:
    :return:
    """

    for archived_file in Path(source_folder).iterdir():

        file_name = Path(archived_file).stem
        extension = Path(archived_file).suffix

        if extension != '.gz':
            continue

        destination_full_path = Path(destination_folder).joinpath(file_name)

        with gzip.open(archived_file, 'rb') as file_in:
            with open(destination_full_path, 'wb') as file_out:
                shutil.copyfileobj(file_in, file_out)
