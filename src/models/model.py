import torch
import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from monai.visualize import blend_images, matshow3d, plot_2d_or_3d_image

from src.models.metrics import METRICS
from src.utils.utils import _CLASSES_METADATA


class ModelSegmentationCT(pl.LightningModule):
    def __init__(self, base_model, loss_function=None):
        super().__init__()
        self.model = base_model
        self.loss_function = loss_function

        self.metrics = METRICS

    def forward(self, data):
        model_output = self.model(data)
        return model_output

    def training_step(self, batch):
        data, target = batch

        prediction = self.model(data)
        loss = self.loss_function(prediction, target)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_index):
        data, target = batch

        prediction = self.model(data)
        val_loss = self.loss_function(prediction, target.float())
        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_index):
        self._compute_metrics(batch, batch_index)

    def _compute_metrics(self, batch, batch_index):
        data, target = batch
        prediction = self.model(data).sigmoid()

        _metrics_to_log = {}

        tp, fp, fn, tn = smp.metrics.get_stats(prediction, target, mode='multilabel', threshold=0.5)

        for _class in _CLASSES_METADATA:
            _tp, _fp, _fn, _tn = tp[:, _class['id'] + 1], fp[:, _class['id'] + 1],\
                                 fn[:, _class['id'] + 1], tn[:, _class['id'] + 1]
            for metric in self.metrics:
                function, reduction = self.metrics[metric]['func'], self.metrics[metric]['reduction']

                metric_value = function(_tp, _fp, _fn, _tn, reduction='macro')

                Y = target[:, _class['id'] + 1].cpu().numpy()
                X = (prediction[:, _class['id'] + 1].cpu().numpy() > 0.5).astype(np.float)

                if not np.count_nonzero(X.sum() + Y.sum()):
                    _metric_value = 1
                else:
                    _metric_value = 2 * np.sum(X * Y) / (X.sum() + Y.sum())

                _metrics_to_log[f"{_class['name']} {metric} custom"] = _metric_value
                print(_metric_value)
                _metrics_to_log[f"{_class['name']} {metric}"] = metric_value

        for metric in self.metrics:
            function, reduction = self.metrics[metric]['func'], self.metrics[metric]['reduction']

            metric_value = function(tp, fp, fn, tn, reduction=reduction)
            _metrics_to_log[f"_{metric}"] = metric_value

        self.log_dict(_metrics_to_log)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

