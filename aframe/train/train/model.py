from typing import Optional, Sequence

import lightning.pytorch as pl
import torch

from aframe.architectures import Architecture
from train.metrics import MatchedFilterLoss
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
        loss_fn: MatchedFilterLoss,
        patience: Optional[int] = None,
        save_top_k_models: int = 10
    ) -> None:
        super().__init__()
        # construct our model up front and record all
        # our hyperparameters to our logdir
        self.model = arch

        self.metric = metric
        self.loss_fn = loss_fn
        self.save_hyperparameters(ignore=["arch", "metric", "loss_fn"])

    def forward(self, X: Tensor) -> Tensor:
        return self.model(X).flip(1)

    def predict(self, X: Tensor) -> Tensor:
        Xh = self.model(X)
        loss, chisq_loss, lowpass_loss = self.loss_fn(Xh, X)
        detstat = 1 / loss
        if chisq_loss is not None:
            detstat /= (1 + chisq_loss**3)**(1 / 6)
        if lowpass_loss is not None:
            detstat -= lowpass_loss
        return detstat.mean(-1)

    def log_loss_component(self, name, value):
        self.log(
            name + "_loss",
            value.mean(),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True
        )

    def training_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        X, psds = batch
        X_hat = self(X)
        loss, chisq_loss, lowpass_loss = self.loss_fn(X_hat, X)

        self.log_loss_component("pearson", loss)
        if chisq_loss is not None:
            self.log_loss_component("chisq", chisq_loss)
            loss = loss * (1 + chisq_loss**3)**(1 / 6)
        if lowpass_loss is not None:
            self.log_loss_component("lowpass", lowpass_loss)
            loss = loss + lowpass_loss

        loss = loss.mean()
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    @property
    def metric_name(self) -> str:
        return f"valid_auroc@{self.metric.metric.max_fpr:0.1e}"

    def validation_step(self, batch, _) -> None:
        shift, (X_bg, psd_bg), (X_fg, psd_fg) = batch
        y_bg = self.predict(X_bg)

        # compute predictions over multiple views of
        # each injection and use their average as our
        # prediction
        num_views, batch, num_ifos, _ = X_fg.shape
        X_fg = X_fg.view(num_views * batch, num_ifos, -1)

        # psd_fg = psd_fg.repeat_interleave(num_views, dim=0)
        y_fg = self.predict(X_fg)
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
