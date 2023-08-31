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
    """

    def __init__(
        self,
        arch: Architecture,
        metric: BinaryAUROC,
        valid_stride: float,
        valid_pool_length: float,
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
        self.validation_step_outputs.append((shift, y_bg, y_fg))

    def on_validation_epoch_end(self) -> None:
        outputs = self.validation_step_outputs

        # break background predictions up into
        # timeseries from different shifts and
        # max pool them to simulate test-time clustering
        shift_vals = list(set([i[0] for i in outputs]))
        shifts = torch.cat([i[0] * torch.ones_like(i[1]) for i in outputs])
        background = torch.cat([i[1] for i in outputs])

        pool_size = int(
            self.hparams.valid_pool_length / self.hparams.valid_stride
        )
        pool_stride = int(pool_size // 2)
        pooled_background = []
        for shift in shift_vals:
            mask = shifts == shift
            preds = background[mask].view(1, 1, -1)
            preds = torch.nn.functional.max_pool1d(
                preds, pool_size, stride=pool_stride, ceil_mode=True
            )
            pooled_background.append(preds[0, 0])
        background = torch.cat(pooled_background)

        # concatenate these with view-averaged foreground
        # predictions to constitute our predicted outputs
        foreground = torch.cat([i[2] for i in outputs])
        y_pred = torch.cat([background, foreground])

        # create 0/1 labels for foreground and background
        y_bg = torch.zeros_like(background)
        y_fg = torch.ones_like(foreground)
        y = torch.cat([y_bg, y_fg])

        # shuffle the prediction and target arrays up
        # front so that constant-output models don't
        # accidently come out perfect
        idx = torch.randperm(len(y_pred))
        y_pred = y_pred[idx]
        y = y[idx]

        # compute and log the auroc
        self.metric.update(y_pred, y)
        auroc = self.metric.compute()
        self.log(self.val_metric, auroc, sync_dist=True)

        # reset our temporary container
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
