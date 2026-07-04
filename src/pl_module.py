import lightning as pl
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryStatScores, BinaryF1Score, BinaryRecall, BinaryPrecision, BinaryAccuracy, \
    BinaryAUROC, BinaryConfusionMatrix, BinaryMatthewsCorrCoef, BinaryJaccardIndex


class FaceMeshLightningModule(pl.LightningModule):
    def __init__(self, model, num_classes: int, optimizer_config: dict, scheduler_config: dict, monitor: str):
        super().__init__()
        self.model = model
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.monitor = monitor
        self.metrics = MetricCollection({
            'accuracy': BinaryAccuracy(),
            'precision': BinaryPrecision(),
            'recall': BinaryRecall(),
            'f1_score': BinaryF1Score(),
            'stat_scores': BinaryStatScores(),
            'matthews_corrcoef': BinaryMatthewsCorrCoef(),
            'jaccard_index': BinaryJaccardIndex(),
            'auroc': BinaryAUROC(),
            'confusion_matrix': BinaryConfusionMatrix(),
        })
        self.metrics_history = []  # List of dicts to store metrics per epoch
        self.save_hyperparameters(ignore=['model'])

    def forward(self, *args, **kwargs):
        logits, _ = self.model(*args, **kwargs)
        return logits

    def predict(self, *args, **kwargs):
        return torch.sigmoid(self.forward(*args, **kwargs))

    def extract_features(self, *args, **kwargs):
        _, features = self.model(*args, **kwargs)
        return features

    def step(self, batch):
        # Accepts dict with at least x and label, optionally edge_index, etc.
        # params = inspect.getfullargspec(self.model.forward)[0]
        # out = torch.flatten(self.forward(**{key: value for key, value in batch.items() if key in params}))
        out = torch.flatten(self.forward(**{key: value for key, value in batch.items()}))
        loss = F.binary_cross_entropy_with_logits(out, batch['y'].float())
        probs = torch.sigmoid(out)
        preds = (probs > 0.5).float()
        return preds, loss

    def training_step(self, batch, batch_idx):
        _, loss = self.step(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch['y'].float()
        preds, loss = self.step(batch)
        self.log('val_loss', loss, prog_bar=True)
        self.metrics(preds, labels.int())
        return loss

    def on_validation_epoch_end(self) -> None:
        epoch_metrics = {'epoch': self.current_epoch}
        prefix = ''
        for name, metric in self.metrics.items():
            value = metric.compute()
            if name == 'confusion_matrix':
                fig, ax = metric.plot()
                self.logger.experiment.add_figure('Confusion matrix', fig, self.current_epoch)
                plt.close(fig)
            elif name == 'stat_scores':
                # tp, fp, tn, fn, sup
                self.log(f'{prefix}{name}/tp', value[0].item() if hasattr(value[0], 'item') else value[0])
                self.log(f'{prefix}{name}/fp', value[1].item() if hasattr(value[1], 'item') else value[1])
                self.log(f'{prefix}{name}/tn', value[2].item() if hasattr(value[2], 'item') else value[2])
                self.log(f'{prefix}{name}/fn', value[3].item() if hasattr(value[3], 'item') else value[3])
                self.log(f'{prefix}{name}/sup', value[4].item() if hasattr(value[4], 'item') else value[4])
                epoch_metrics[f'{prefix}{name}/tp'] = value[0].item() if hasattr(value[0], 'item') else value[0]
                epoch_metrics[f'{prefix}{name}/fp'] = value[1].item() if hasattr(value[1], 'item') else value[1]
                epoch_metrics[f'{prefix}{name}/tn'] = value[2].item() if hasattr(value[2], 'item') else value[2]
                epoch_metrics[f'{prefix}{name}/fn'] = value[3].item() if hasattr(value[3], 'item') else value[3]
                epoch_metrics[f'{prefix}{name}/sup'] = value[4].item() if hasattr(value[4], 'item') else value[4]
            else:
                self.log(f"{prefix}{name}", value, prog_bar=True)
                epoch_metrics[f"{prefix}{name}"] = value.item() if hasattr(value, 'item') else value

            metric.reset()

        # Append metrics for this epoch to history
        self.metrics_history.append(epoch_metrics)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_config)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **self.scheduler_config)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': self.monitor}
