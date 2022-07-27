import torch
import pytorch_lightning as pl


class ModelSegmentationCT(pl.LightningModule):
    def __init__(self, base_model, loss_function=None):
        super().__init__()
        self.model = base_model
        self.loss_function = loss_function

    def forward(self, data):
        model_output = self.model(data)
        return model_output

    def training_step(self, batch):
        data, target = batch

        prediction = self.model(data)
        loss = self.loss_function(prediction, target.float())

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_index):
        data, target = batch

        prediction = self.model(data)
        val_loss = self.loss_function(prediction, target.float())
        self.log("val_loss", val_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
