from collections import defaultdict
from typing import Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from monai.networks.nets import UNETR
import segmentation_models_pytorch as smp
from tqdm import tqdm

from src.config import Config
from src.models.model import ModelSegmentationCT
from src.data.datasets import AMOS22Volumes
from src.utils.utils import _CLASSES_METADATA
from monai.visualize import blend_images, matshow3d, plot_2d_or_3d_image


def init_model(config: Config) -> Union[ModelSegmentationCT, None]:

    if config.model == 'UNET':
        model = ModelSegmentationCT.load_from_checkpoint(
            checkpoint_path=config.checkpoint_path,
            base_model=smp.Unet(
                encoder_name='resnet34',
                classes=config.number_of_classes * config.channels,
                in_channels=config.channels
            )
        )
    elif config.model == 'UNETR':
        model = ModelSegmentationCT.load_from_checkpoint(
            checkpoint_path=config.checkpoint_path,
            base_model=UNETR(
                in_channels=config.channels,
                out_channels=config.number_of_classes * config.channels,
                img_size=config.image_shape,
                spatial_dims=2
            )
        )
    else:
        print(f'{config.model} not found')
        return None

    return model


def compute_metrics(prediction, target, _metrics_to_log):

    for _class in _CLASSES_METADATA:

        Y = target[:, _class['id'] + 1]
        X = prediction[_class['id'] + 1]

        if not np.count_nonzero(X.sum() + Y.sum()):
            metric_value = 1
        else:
            metric_value = 2 * np.sum(X * Y) / (X.sum() + Y.sum())

        _metrics_to_log[_class['name']]['metrics'].append(metric_value)
        _metrics_to_log[_class['name']]['values'].append(Y.sum())

    return _metrics_to_log


def postprocess(model_output, threshold=0.5):
    return (model_output.detach().cpu().permute(1, 0, 2, 3).sigmoid() > threshold).numpy().astype(np.float32)


def reshape_data_as_channels(input_data, channels):
    length, _, width, height = input_data.shape
    crop_size = length - length % channels

    cropped_data = input_data[:crop_size].reshape(-1, channels, width, height)
    tail_data = input_data[-channels:].reshape(-1, channels, width, height)

    prepared_data = torch.concat((cropped_data, tail_data), dim=0)
    return prepared_data


def reverse_reshape_data_as_channels(output_data, channels, original_length, num_classes=16):
    crop_size = original_length - original_length % channels

    batch_size, _, width, height = output_data.shape
    output_data = output_data.reshape(batch_size, num_classes, channels, width, height)\
        .permute(0, 2, 1, 3, 4).reshape(-1, num_classes, width, height)

    result_output_data = output_data[:crop_size]

    if (original_length % channels) != 0:
        tail_data = output_data[-(original_length % channels):]
        result_output_data = torch.concat((result_output_data, tail_data), dim=0)

    return result_output_data


def inference(input_data, batch_size, model, channels=None):
    num_steps = len(input_data) // batch_size

    start = 0
    end = min(batch_size, len(input_data))

    sub_data = []

    for i in range(num_steps + 1):

        if start == end:
            continue

        sub_input_data = input_data[start:end]

        if channels is not None:
            sub_input_data = reshape_data_as_channels(sub_input_data, channels)

        sub_output_data = model(sub_input_data)

        if channels is not None:
            original_length = end - start
            sub_output_data = reverse_reshape_data_as_channels(sub_output_data, channels, original_length)

        sub_data.append(sub_output_data)

        start += batch_size
        end = min(len(input_data), end + batch_size)

    output = torch.concat(sub_data, dim=0)
    assert output.shape[0] == input_data.shape[0], f'{output.shape} != {input_data.shape}'

    return output


def test(config: Config):

    device = torch.device(config.device)
    dataset = AMOS22Volumes(path_to_data='/home/vpavlishen/data/vpavlishen/foloct/AMOS22/raw/AMOS22', stage='test',
                            config=config)
    dataloader = DataLoader(dataset=dataset, num_workers=config.num_workers, shuffle=False, batch_size=None)

    model = init_model(config)

    _metrics_to_log = defaultdict(lambda: {'metrics': [], 'values': []})
    with torch.no_grad():
        model.eval().to(device)

        for volume, target in tqdm(dataloader):
            input_data = volume.to(device)

            model_output = inference(input_data, config.sub_batch_size, model,
                                     channels=config.channels if config.channels != 1 else None)
            output_detached = postprocess(model_output)
            _metrics_to_log = compute_metrics(output_detached, target, _metrics_to_log)

            del volume, input_data, model_output
            torch.cuda.empty_cache()

    for metric in _metrics_to_log:
        print(f"{metric}: \t {np.array(_metrics_to_log[metric]['metrics']).mean():.4f}")


if __name__ == '__main__':

    # Exp1 UNET "/home/vpavlishen/lightning_logs/version_38/checkpoints/epoch=29-step=4290.ckpt"
    # Exp1 UNETR "/home/vpavlishen/lightning_logs/version_59/checkpoints/epoch=29-step=20850.ckpt"
    # Exp1 UNETR  256 "/home/vpavlishen/lightning_logs/version_77/checkpoints/epoch=149-step=208500.ckpt"

    # Exp2 UNET "/home/vpavlishen/lightning_logs/version_7/checkpoints/epoch=89-step=8640.ckpt"
    # Exp2 UNETR "/home/vpavlishen/lightning_logs/version_61/checkpoints/epoch=79-step=14400.ckpt"

    # Exp2 UNETR 256x256 "/home/vpavlishen/lightning_logs/version_93/checkpoints/epoch=39-step=7200.ckpt"
    # Exp2 UNETR 128x128 /home/vpavlishen/lightning_logs/version_90/checkpoints/epoch=149-step=27000.ckpt

    # Exp3 3 channels UNET "/home/vpavlishen/lightning_logs/version_54/checkpoints/epoch=79-step=12960.ckpt"
    # Exp3 3 channels UNETR "/home/vpavlishen/lightning_logs/version_63/checkpoints/epoch=79-step=40480.ckpt"

    # Exp3 5 channels UNET "/home/vpavlishen/lightning_logs/version_56/checkpoints/epoch=9-step=2270.ckpt"
    # Exp3 5 channels UNETR "/home/vpavlishen/lightning_logs/version_67/checkpoints/epoch=79-step=40480.ckpt"

    # Exp3 10 channels UNET "/home/vpavlishen/lightning_logs/version_58/checkpoints/epoch=19-step=6540.ckpt"
    # Exp3 10 channels UNETR "/home/vpavlishen/lightning_logs/version_69/checkpoints/epoch=79-step=111200.ckpt"

    test_config = Config(
        sub_batch_size=200,
        num_workers=4,
        image_shape=(128, 128),
        patch_shape=(-1, 128, 128),
        crop_shape=(800, 128, 128),
        model='UNET',
        device='cuda',
        checkpoint_path="/home/vpavlishen/lightning_logs/version_95/checkpoints/epoch=109-step=19800.ckpt",
        channels=1
    )

    test(test_config)
