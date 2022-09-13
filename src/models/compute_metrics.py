from collections import defaultdict
from typing import Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from monai.networks.nets import UNETR
import segmentation_models_pytorch as smp

from src.config import Config
from src.models.model import ModelSegmentationCT
from src.data.datasets import AMOS22Volumes
from src.utils.utils import _CLASSES_METADATA
from monai.visualize import blend_images, matshow3d, plot_2d_or_3d_image


def init_model(config: Config) -> Union[ModelSegmentationCT, None]:

    if config.model == 'UNET':
        model = ModelSegmentationCT.load_from_checkpoint(
            checkpoint_path=config.checkpoint_path,
            base_model=smp.Unet('resnet34', classes=config.number_of_classes, in_channels=config.channels)
        )
    elif config.model == 'UNETR':
        model = ModelSegmentationCT.load_from_checkpoint(
            checkpoint_path=config.checkpoint_path,
            base_model=UNETR(
                in_channels=config.channels, out_channels=config.number_of_classes,
                img_size=config.image_shape, spatial_dims=2
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


def inference(input_data, batch_size, model):
    num_steps = len(input_data) // batch_size

    start = 0
    end = min(batch_size, len(input_data))

    sub_data = []

    for i in range(num_steps + 1):

        if start == end:
            continue

        sub_input_data = input_data[start:end]
        sub_output_data = model(sub_input_data)

        sub_data.append(sub_output_data)

        start += batch_size
        end = min(len(input_data), end + batch_size)

    output = torch.concat(sub_data, dim=0)
    assert output.shape[0] == input_data.shape[0]

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

        for i, (volume, target) in enumerate(dataloader):
            input_data = volume.to(device)

            model_output = inference(input_data, config.sub_batch_size, model)
            output_detached = postprocess(model_output)
            _metrics_to_log = compute_metrics(output_detached, target, _metrics_to_log)

            del volume, input_data, model_output
            torch.cuda.empty_cache()

    for metric in _metrics_to_log:
        print(metric, np.array(_metrics_to_log[metric]['metrics']).mean(),
              np.array(_metrics_to_log[metric]['values']).mean())


if __name__ == '__main__':

    # Exp1 UNET "/home/vpavlishen/lightning_logs/version_38/checkpoints/epoch=29-step=4290.ckpt"
    # Exp1 UNETR "/home/vpavlishen/lightning_logs/version_59/checkpoints/epoch=29-step=20850.ckpt"
    # Exp1 UNETR  256 "/home/vpavlishen/lightning_logs/version_77/checkpoints/epoch=149-step=208500.ckpt"

    # Exp2 UNET "/home/vpavlishen/lightning_logs/version_7/checkpoints/epoch=89-step=8640.ckpt"
    # Exp2 UNETR "/home/vpavlishen/lightning_logs/version_61/checkpoints/epoch=79-step=14400.ckpt"

    # Exp3 3 channels UNET "/home/vpavlishen/lightning_logs/version_54/checkpoints/epoch=79-step=12960.ckpt"
    # Exp3 3 channels UNETR "/home/vpavlishen/lightning_logs/version_63/checkpoints/epoch=79-step=40480.ckpt"

    # Exp3 5 channels UNET "/home/vpavlishen/lightning_logs/version_56/checkpoints/epoch=9-step=2270.ckpt"
    # Exp3 5 channels UNETR "/home/vpavlishen/lightning_logs/version_67/checkpoints/epoch=79-step=40480.ckpt"

    # Exp3 10 channels UNET "/home/vpavlishen/lightning_logs/version_58/checkpoints/epoch=19-step=6540.ckpt"
    # Exp3 10 channels UNETR "/home/vpavlishen/lightning_logs/version_69/checkpoints/epoch=79-step=111200.ckpt"

    test_config = Config(
        sub_batch_size=300, num_workers=4, image_shape=(256, 256), patch_shape=(-1, 256, 256), model='UNETR',
        device='cuda', checkpoint_path="/home/vpavlishen/lightning_logs/version_77/checkpoints/epoch=149-step=208500.ckpt"
    )

    test(test_config)
