import os
import glob
from pathlib import Path
from typing import NoReturn

import numpy as np
from tqdm import tqdm

TRAIN_SIZE = 0.75
VAL_SIZE = 0.15
TEST_SIZE = 0.1
RANDOM_SEED = 42


def split_dataset(root_dir: str, output_dir: str) -> NoReturn:
    """
    Split dataset into train, val, test
    :param root_dir with data
    :param output_dir: dir for data saving
    :return:
    """

    all_data_files = glob.glob(str(Path(root_dir).joinpath('imagesTr', '*.nii.gz')))

    train_size = int(len(all_data_files) * TRAIN_SIZE)
    val_size = int(len(all_data_files) * VAL_SIZE)
    test_size = int(len(all_data_files) * TEST_SIZE)

    assert train_size + val_size + test_size == len(all_data_files), \
        f'{len(all_data_files)}, {train_size}, {val_size}, {test_size}'

    np.random.seed(RANDOM_SEED)
    all_data_files_shuffled = np.random.permutation(all_data_files)

    train_files, val_files, test_files = np.split(all_data_files_shuffled, [train_size, train_size + val_size])

    for split_files, split_name in zip((train_files, val_files, test_files), ('train', 'val', 'test')):
        for source_file in tqdm(split_files):

            dir_for_saving = Path(output_dir).joinpath(split_name)
            image_dir_for_saving = dir_for_saving.joinpath('imagesTr')
            label_dir_for_saving = dir_for_saving.joinpath('labelsTr')

            if not os.path.exists(image_dir_for_saving):
                os.makedirs(image_dir_for_saving)

            if not os.path.exists(label_dir_for_saving):
                os.makedirs(label_dir_for_saving)

            file_name = Path(source_file).name
            source_label = source_file.replace('imagesTr', 'labelsTr')

            os.rename(source_file, image_dir_for_saving.joinpath(file_name))
            os.rename(source_label, label_dir_for_saving.joinpath(file_name))


if __name__ == '__main__':
    split_dataset('data/vpavlishen/AMOS22/', 'data/vpavlishen/processed/AMOS22_splitted')