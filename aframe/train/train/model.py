import logging
from typing import Optional, Sequence

import lightning.pytorch as pl
import torch
from torchmetrics.classification import BinaryAUROC

from aframe.architectures import Architecture


class Aframe(pl.LightningModule):
    """
    Args:
        arch: Architecture to train on
        data_dir: Path to data
    """

    def __init__(
        self,
        arch: Architecture,
        metric: BinaryAUROC,
        patience: Optional[int] = None,
    ) -> None:
        super().__init__()
        # construct our model up front and record all
        # our hyperparameters to our logdir
        self.model = arch
        self.metric = metric
        self.save_hyperparameters(ignore=["arch", "metric"])

        # set up a console logger to keep track
        # of some of our more intensive tasks
        self._logger = logging.getLogger("Aframe")
        self.validation_step_outputs = []

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        X, y = batch
        y_hat = self(X)
        loss = self.compute_loss(y_hat, y).mean()
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def compute_loss(
        self, y_hat: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        return torch.nn.functional.binary_cross_entropy_with_logits(y_hat, y)

    @property
    def val_metric(self) -> str:
        return f"valid_auroc@{self.metric.max_fpr:0.1e}"

    def validation_step(self, batch, _) -> None:
        shift, X_bg, X_inj = batch
        y_bg = self(X_bg)[:, 0]

        num_views, batch, num_ifos, _ = X_inj.shape
        X_inj = X_inj.view(num_views * batch, num_ifos, -1)
        y_fg = self(X_inj)
        y_fg = y_fg.view(num_views, batch)
        y_fg = y_fg.mean(0)
        self.validation_step_outputs.append((shift, y_bg, y_fg))

    def on_validation_epoch_end(self) -> None:
        # TODO: use shift to do pooling on background
        outputs = self.validation_step_outputs
        background = torch.cat([i[1] for i in outputs])
        foreground = torch.cat([i[2] for i in outputs])
        y_pred = torch.cat([background, foreground])

        y_bg = torch.zeros_like(background)
        y_fg = torch.ones_like(foreground)
        y = torch.cat([y_bg, y_fg])

        # shuffle the prediction and target arrays up
        # front so that constant-output models don't
        # accidently come out perfect
        idx = torch.randperm(len(y_pred))
        y_pred = y_pred[idx]
        y = y[idx]
        auroc = self.metric(y_pred, y)
        self.log(self.val_metric, auroc, sync_dist=True)

        self.validation_step_outputs.clear()

    def configure_callbacks(self) -> Sequence[pl.Callback]:
        # TODO: make top_k a param
        checkpoint = pl.callbacks.ModelCheckpoint(
            monitor=self.val_metric,
            save_top_k=10,
            save_last=True,
            auto_insert_metric_name=False,
            mode="max",
        )
        callbacks = [checkpoint]
        if self.hparams.patience is not None:
            early_stop = pl.callbacks.EarlyStop(
                monitor=self.val_metric,
                patience=self.hparams.patience,
                mode="max",
                min_delta=0.00,
            )
            callbacks.append(early_stop)
        return callbacks
