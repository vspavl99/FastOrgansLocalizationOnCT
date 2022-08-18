import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from src.models.metrics import METRICS


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
        prediction = self.model(data)

        _metrics_to_log = {}

        tp, fp, fn, tn = smp.metrics.get_stats(prediction, target, mode='multilabel', threshold=0.5)

        for metric in self.metrics:
            function, reduction = self.metrics[metric]['func'], self.metrics[metric]['reduction']

            _metrics_to_log[metric] = function(tp, fp, fn, tn, reduction=reduction)

        self.log_dict(_metrics_to_log)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

