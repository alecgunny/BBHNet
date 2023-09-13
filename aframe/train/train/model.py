from typing import Optional, Sequence

import lightning.pytorch as pl
import torch

from aframe.architectures import Architecture
from train.validation import TimeSlideAUROC

Tensor = torch.Tensor


class Aframe(pl.LightningModule):
    """
    Args:
        arch: Architecture to train on
        metric: BinaryAUROC metric used for evaluation
        valid_stride:
            Time between inference evaluations during
            validation, in seconds
        valid_pool_length:
            Length of time over which to max pool background
            predictions during validation, in seconds
        patience:
            Number of epochs to wait for an increase in
            validation AUROC before terminating training.
            If left as `None`, will never terminate
            training early.
    """

    def __init__(
        self,
        arch: Architecture,
        metric: TimeSlideAUROC,
        patience: Optional[int] = None,
        save_top_k_models: int = 10
    ) -> None:
        super().__init__()
        # construct our model up front and record all
        # our hyperparameters to our logdir
        self.model = arch
        self.metric = metric
        self.save_hyperparameters(ignore=["arch", "metric"])

    def forward(self, X: Tensor) -> Tensor:
        return self.model(X)

    def training_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
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

    def compute_loss(self, y_hat: Tensor, y: Tensor) -> Tensor:
        return torch.nn.functional.binary_cross_entropy_with_logits(y_hat, y)

    @property
    def metric_name(self) -> str:
        return f"valid_auroc@{self.metric.metric.max_fpr:0.1e}"

    def validation_step(self, batch, _) -> None:
        shift, X_bg, X_inj = batch
        y_bg = self(X_bg)[:, 0]

        # compute predictions over multiple views of
        # each injection and use their average as our
        # prediction
        num_views, batch, num_ifos, _ = X_inj.shape
        X_inj = X_inj.view(num_views * batch, num_ifos, -1)
        y_fg = self(X_inj)
        y_fg = y_fg.view(num_views, batch)
        y_fg = y_fg.mean(0)

        # include the shift associated with this data
        # in our outputs to reconstruct background
        # timeseries at aggregation time
        self.metric.update(shift, y_bg, y_fg)

        # lightning will take care of updating then
        # computing the metric at the end of the
        # validation epoch
        self.log(
            self.metric_name,
            self.metric,
            on_step=True,
            on_epoch=True,
            sync_dist=True
        )

    def configure_callbacks(self) -> Sequence[pl.Callback]:
        checkpoint = pl.callbacks.ModelCheckpoint(
            monitor=self.metric_name,
            save_top_k=self.hparams.save_top_k_models,
            save_last=True,
            auto_insert_metric_name=False,
            mode="max",
        )
        callbacks = [checkpoint]
        if self.hparams.patience is not None:
            early_stop = pl.callbacks.EarlyStop(
                monitor=self.metric_name,
                patience=self.hparams.patience,
                mode="max",
                min_delta=0.00,
            )
            callbacks.append(early_stop)
        return callbacks
