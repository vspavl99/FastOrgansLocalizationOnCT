from typing import NoReturn
import os
import glob
from pathlib import Path

import numpy as np
from tqdm import tqdm
from monai.data import ImageDataset


def slice_saving(volume: np.ndarray, segmentation: np.ndarray, file_name: str,
                 volume_index: int, output_dir: str) -> NoReturn:
    """
    Iterate over 3D array and save slices on disk
    :param volume: volume as numpy array
    :param segmentation: segmentation as numpy array
    :param file_name: base file name
    :param volume_index: index of volume in dataset
    :param output_dir: directory for saving slices
    :return:
    """

    num_samples = volume.shape[2]

    for slice_num in range(num_samples):
        volume_slice = volume[:, :, slice_num].numpy()
        segmentation_slice = segmentation[:, :, slice_num].numpy().astype(np.uint8)

        slice_name = str(Path(file_name).name).replace('.nii.gz', f'_{slice_num}')
        output_volume_slice_path = Path(output_dir).joinpath('imagesTr', str(volume_index), slice_name)
        output_segmentation_slice_path = Path(output_dir).joinpath('labelsTr', str(volume_index), slice_name)

        if not os.path.exists(str(output_volume_slice_path.parent)):
            os.makedirs(str(output_volume_slice_path.parent))

        if not os.path.exists(str(output_segmentation_slice_path.parent)):
            os.makedirs(str(output_segmentation_slice_path.parent))

        np.savez_compressed(str(output_volume_slice_path), volume_slice)
        np.savez_compressed(str(output_segmentation_slice_path), segmentation_slice)


def split_volumes_into_2d_images(root_dir: str, output_dir: str) -> NoReturn:
    """
    Extract slices from 3D volume and save them as 2d image on disk
    :param root_dir: directory with volumes and labels
    :param output_dir: directory where 2d images will be saved
    :return:
    """

    path_images_template = str(Path(root_dir).joinpath('imagesTr', '*.nii.gz'))
    path_labels_template = str(Path(root_dir).joinpath('labelsTr', '*.nii.gz'))

    images = sorted(glob.glob(path_images_template))
    labels = sorted(glob.glob(path_labels_template))

    dataset = ImageDataset(image_files=images, seg_files=labels)

    for index in tqdm(range(len(dataset))):
        volume, segmentation = dataset[index]
        volume_name = dataset.image_files[index]

        volume_shape = volume.shape

        min_dimension = np.argmin(np.array(volume_shape))
        if min_dimension != 2:
            volume, segmentation = volume.moveaxis(min_dimension, 2), segmentation.moveaxis(min_dimension, 2)

        slice_saving(volume=volume, segmentation=segmentation, file_name=volume_name,
                     volume_index=index, output_dir=output_dir)


if __name__ == '__main__':
    split_volumes_into_2d_images(root_dir='data/vpavlishen/AMOS22', output_dir='data/vpavlishen/processed/AMOS22')
