import numpy as np
import segmentation_models_pytorch as smp

from src.config import TrainConfig
from src.data.dataset import CTDatasetPatches
from src.data.base_datasets import AMOS22
from src.models.model import ModelSegmentationCT

METRICS = {
    'iou_score': {
        'values': [],
        'func': smp.metrics.iou_score,
        'reduction': 'micro'
    },
    'f1_score': {
        'values': [],
        'func': smp.metrics.f1_score,
        'reduction': 'micro'
    },
    'accuracy': {
        'values': [],
        'func': smp.metrics.accuracy,
        'reduction': 'micro'
    },
    'recall': {
        'values': [],
        'func': smp.metrics.recall,
        'reduction': 'micro-imagewise'
    },
}


if __name__ == '__main__':
    device = 'cuda'

    model = ModelSegmentationCT.load_from_checkpoint(
        checkpoint_path="/home/vpavlishen/lightning_logs/version_7/checkpoints/epoch=89-step=8640.ckpt",
        base_model=smp.Unet('resnet34', encoder_weights='imagenet', classes=16, in_channels=1)
    )
    model.eval()
    model.to(device)

    train_config = TrainConfig(batch_size=2, num_workers=0)

    dataset = AMOS22(path_to_data='/home/vpavlishen/data/vpavlishen/AMOS22')
    data = CTDatasetPatches(dataset=dataset, config=train_config)
    data.setup()

    val_dataloader = data.val_dataloader()
    metrics = METRICS

    for image, target in val_dataloader:
        preds = model(image.to(device)).detach().cpu()

        tp, fp, fn, tn = smp.metrics.get_stats(preds, target, mode='multilabel', threshold=0.5)

        for metric in metrics:
            function, reduction = metrics[metric]['func'], metrics[metric]['reduction']
            metrics[metric]['values'].append(function(tp, fp, fn, tn, reduction=reduction).item())

    for metric in metrics:
        print(f'{metric} {np.array(metrics[metric]).mean()}')
