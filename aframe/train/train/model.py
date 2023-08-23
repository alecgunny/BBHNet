import glob
import logging
import os
from typing import Optional, Sequence

import h5py
import lightning.pytorch as pl
import torch
from torchmetrics.classification import BinaryAUROC
from train import augmentations as aug
from train.validation import TimeSlideDataset, ZippedDataset

from aframe.architectures import Architecture
from aframe.architectures.preprocessing import PsdEstimator
from ml4gw.dataloading import ChunkedDataset
from ml4gw.distributions import PowerLaw
from ml4gw.transforms import Whiten
from ml4gw.utils.slicing import unfold_windows


class Aframe(pl.LightningModule):
    """
    Args:
        arch: Architecture to train on
        data_dir: Path to data
    """

    def __init__(
        self,
        arch: Architecture,
        data_dir: str,
        ifos: Sequence[str],
        valid_frac: float,
        batch_size: int,
        snr_thresh: float,
        max_snr: float,
        snr_alpha: float,
        # data args
        kernel_length: float,
        psd_length: float,
        fduration: float,
        highpass: float,
        fftlength: Optional[float] = None,
        # optimization_args
        max_lr: float = 1e-3,
        weight_decay: float = 0,
        lr_ramp_epochs: Optional[int] = None,
        patience: Optional[int] = None,
        # augmentation args
        waveform_prob: float = 0.5,
        swap_frac: float = 0.0,
        mute_frac: float = 0.0,
        trigger_pad: float = 0,
        # validation args
        valid_stride: Optional[float] = None,
        num_valid_views: int = 5,
        max_fpr: float = 1e-3,
        valid_livetime: float = (3600 * 12),
    ) -> None:
        super().__init__()
        # construct our model up front and record all
        # our hyperparameters to our logdir
        self.model = arch
        self.save_hyperparameters(ignore=["arch"])

        # set up a console logger to keep track
        # of some of our more intensive tasks
        self._logger = logging.getLogger("Aframe")

        # infer the sample rate from the data
        with h5py.File(self.train_fnames[0], "r") as f:
            self.sample_rate = 1 / f[ifos[0]].attrs["dx"]
        self._logger.info(f"Inferred sample rate {self.sample_rate}")

        # set up some of the modules we'll need for
        # 1. Preprocessing
        fftlength = fftlength or kernel_length + fduration
        self.psd_estimator = PsdEstimator(
            kernel_length + fduration,
            self.sample_rate,
            fftlength,
            fast=highpass is not None,
        )
        self.whitener = Whiten(fduration, self.sample_rate, highpass)

        # 2. Data augmentation
        self.inverter = aug.SignalInverter(0.5)
        self.reverser = aug.SignalReverser(0.5)
        self.snr_rescaler = aug.SnrRescaler(self.sample_rate, highpass)

        # 3. Validation
        self.auroc = BinaryAUROC(max_fpr=max_fpr)

        # now actually load in our training signal data.
        # Training background will be loaded on the fly
        # in chunks during training.
        self._logger.info("Loading training waveforms")
        waveforms = self.load_waveforms("train")
        self._logger.info(f"Loaded {len(waveforms)} training waveforms")
        self.waveform_sampler = aug.WaveformSampler(
            ifos,
            self.sample_rate,
            waveform_prob,
            mute_frac,
            swap_frac,
            self.pad_size,
            highpass=highpass,
            snr_sampler=PowerLaw(snr_thresh, max_snr, snr_alpha),
            cross=waveforms[:, 0],
            plus=waveforms[:, 1],
        )

        self.validation_step_outputs = []

    @property
    def pad_size(self) -> int:
        return int(self.hparams.trigger_pad * self.sample_rate)

    @property
    def train_fnames(self) -> Sequence[str]:
        fnames = glob.glob(f"{self.hparams.data_dir}/background/*.hdf5")
        return sorted(fnames)[:-1]

    @property
    def valid_fnames(self) -> Sequence[str]:
        fnames = glob.glob(f"{self.hparams.data_dir}/background/*.hdf5")
        return sorted(fnames)[-1:]

    @property
    def steps_per_epoch(self) -> int:
        waveforms_per_batch = (
            self.hparams.batch_size * self.hparams.waveform_prob
        )
        num_waveforms = self.waveform_sampler.length
        total_batches = int(4 * num_waveforms / waveforms_per_batch)
        return int(total_batches / len(self.trainer.device_ids))

    @property
    def sample_length(self) -> float:
        return (
            self.hparams.kernel_length
            + self.hparams.fduration
            + self.hparams.psd_length
        )

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

    def on_after_batch_transfer(self, batch, _):
        if self.trainer.training:
            # if we're training, perform random augmentations
            # on input data and use it to impact labels
            y = torch.zeros((batch.size(0), 1), device=batch.device)
            batch = self.augment(batch, y)
        elif self.trainer.validating or self.trainer.sanity_checking:
            # otherwise, if we're validating, unfold the background
            # data into a batch of overlapping kernels now that
            # we're on the GPU so that we're not transferring as
            # much data from CPU to GPU
            background, [signals] = batch
            sample_size = int(self.sample_length * self.sample_rate)
            stride_size = int(self.hparams.valid_stride * self.sample_rate)
            background = unfold_windows(
                background, sample_size, stride=stride_size
            )
            batch = (background, signals)
        return batch

    def augment(
        self, X: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        X, psds = self.psd_estimator(X)

        X = self.inverter(X)
        X = self.reverser(X)

        X, mask = self.waveform_sampler(X, psds)
        y[mask] += 1

        X = self.whitener(X, psds)
        return X, y

    @property
    def val_metric(self) -> str:
        return f"valid_auroc@{self.hparams.max_fpr:0.1e}"

    def validation_step(self, batch, _) -> None:
        background, signals = batch
        X, psd = self.psd_estimator(background)
        X_bg = self.whitener(X, psd)
        y_bg = self(X_bg)[:, 0]

        step = int(len(X) / len(signals))

        # sometimes at the end of a segment,
        # there won't be enough background
        # kernels and so we'll have to inject
        # our signals on overlapping data and
        # ditch some at the end
        if not step:
            signals = signals[: len(X)]
        else:
            X = X[::step][: len(signals)]
            psd = psd[::step][: len(signals)]

        # create `num_view` instances of the injection on top of
        # the background, each showing a different, overlapping
        # portion of the signal
        kernel_size = X.size(-1)
        center = signals.size(-1) // 2

        step = kernel_size + 2 * self.pad_size
        step /= self.hparams.num_valid_views - 1
        X_inj = []
        for i in range(self.hparams.num_valid_views):
            start = center + self.pad_size - int(i * step)
            stop = start + kernel_size
            injected = X + signals[:, :, int(start) : int(stop)]
            injected = self.whitener(injected, psd)
            X_inj.append(injected)
        X_inj = torch.cat(X_inj, dim=0)

        y_fg = self(X_inj)
        y_fg = y_fg.reshape(self.hparams.num_valid_views, -1)
        y_fg = y_fg.mean(dim=0)

        self.validation_step_outputs.append((y_bg, y_fg))

    def on_validation_epoch_end(self) -> None:
        # TODO: add pooling on background
        outputs = self.validation_step_outputs
        background = torch.cat([i[0] for i in outputs])
        foreground = torch.cat([i[1] for i in outputs])
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
        auroc = self.auroc(y_pred, y)
        self.log(self.val_metric, auroc, sync_dist=True)

        self.validation_step_outputs.clear()

    def load_waveforms(self, subset: str) -> torch.Tensor:
        with h5py.File(f"{self.hparams.data_dir}/signals.h5", "r") as f:
            num_signals = len(f["signals"])
            num_valid_signals = int(self.hparams.valid_frac * num_signals)
            if subset == "train":
                num_signals -= num_valid_signals
                slc = slice(None, num_signals)
            elif subset == "val":
                slc = slice(-num_valid_signals, None)
            else:
                raise ValueError(f"Unknown waveform subset {subset}")

            return torch.Tensor(f["signals"][slc])

    @torch.no_grad()
    def project_waveforms(self, waveforms, dec, psi, phi, background):
        # move the density estimator window to cpu
        # so we can compute the PSD without having
        # to move a potentially multi-GB timeseries to GPU
        device = self.psd_estimator.spectral_density.window.device
        self.psd_estimator.spectral_density.to("cpu")
        psd = self.psd_estimator.spectral_density(background.double())

        # now move the PSD and the window back to the
        # desired device
        psd = psd.to(device)
        self.psd_estimator.spectral_density.to(device)

        # now do the projection iteratively on the GPU in batches
        batch_size = self.hparams.batch_size * 4
        num_batches = (len(waveforms) - 1) // batch_size + 1
        responses = []
        for i in range(num_batches):
            slc = slice(i * batch_size, (i + 1) * batch_size)
            params = [i[slc].to(device) for i in [dec, phi, psi]]
            response = self.waveform_sampler.project(
                *params,
                snrs=self.hparams.snr_thresh,
                psds=psd,
                cross=waveforms[slc, 0].to(device),
                plus=waveforms[slc, 1].to(device),
            )
            responses.append(response.cpu())
        return torch.cat(responses, dim=0)

    def val_dataloader(self) -> ZippedDataset:
        # how much larger do we want to make validation
        # batches than the batch size specified for
        # training
        val_batch_factor = 4

        # load in the validation background file
        # and create a dataset that iterates through
        # it while timeshifting the IFOs
        background = []
        with h5py.File(self.valid_fnames[0], "r") as f:
            for ifo in self.hparams.ifos:
                background.append(torch.Tensor(f[ifo][:]))
        background = torch.stack(background)
        background_dataset = TimeSlideDataset(
            background,
            self.sample_rate,
            self.sample_length,
            self.hparams.valid_stride,
            val_batch_factor * self.hparams.batch_size,
            self.hparams.valid_livetime,
            shift=1,
        )

        # load in our validation waveforms and project
        # them to interferometer responses, then rescale
        # them so that the minimum snr is self.hparams.snr_thresh
        waveforms = self.load_waveforms("val")
        num_waveforms = len(waveforms)
        with h5py.File(f"{self.hparams.data_dir}/signals.h5", "r") as f:
            dec = torch.Tensor(f["dec"][-num_waveforms:])
            psi = torch.Tensor(f["psi"][-num_waveforms:])
            phi = torch.Tensor(f["ra"][-num_waveforms:])
        responses = self.project_waveforms(
            waveforms, dec, psi, phi, background
        )

        # Figure out how many batches of background
        # we're going to go through, then batch the
        # signals so that they're spaced evenly
        # throughout all those batches.
        # TODO: should we just assign this to
        # __len__ of TimeSlideDataset?
        num_val_batches = int(
            self.hparams.valid_livetime
            / self.hparams.valid_stride
            / self.hparams.batch_size
            / val_batch_factor
        )
        signal_batch_size = (num_waveforms - 1) // num_val_batches + 1
        signal_dataset = torch.utils.data.TensorDataset(responses)
        signal_loader = torch.utils.data.DataLoader(
            signal_dataset,
            batch_size=signal_batch_size,
            shuffle=False,
            pin_memory=False,
        )

        dataset = ZippedDataset(background_dataset, signal_loader)
        return dataset

    def train_dataloader(self) -> ChunkedDataset:
        chunks_per_epoch = 80
        reads_per_chunk = 10
        num_workers = 1

        batches_per_chunk = int(self.steps_per_epoch / chunks_per_epoch)
        if len(self.trainer.device_ids) > 1:
            pin_memory = True
            rank = int(os.environ["LOCAL_RANK"])
            device_id = self.trainer.device_ids[rank]
            device = f"cuda:{device_id}"
        elif isinstance(
            self.trainer.accelerator, pl.accelerators.CUDAAccelerator
        ):
            device = f"cuda:{self.trainer.device_ids[0]}"
            pin_memory = True
        else:
            device = "cpu"
            pin_memory = False

        return ChunkedDataset(
            self.train_fnames,
            channels=self.hparams.ifos,
            kernel_length=self.sample_length,
            sample_rate=self.sample_rate,
            batch_size=self.hparams.batch_size,
            reads_per_chunk=reads_per_chunk,
            chunk_length=1024,
            batches_per_chunk=batches_per_chunk,
            chunks_per_epoch=chunks_per_epoch,
            num_workers=num_workers,
            device=device,
            pin_memory=pin_memory,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams.max_lr,
            weight_decay=self.hparams.weight_decay,
        )

        if self.hparams.lr_ramp_epochs is None:
            pct_start = 0.3
        else:
            pct_start = self.hparams.lr_ramp_epochs / self.trainer.max_epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.max_lr,
            epochs=self.trainer.max_epochs,
            pct_start=pct_start,
            steps_per_epoch=self.steps_per_epoch,
            anneal_strategy="cos",
        )
        return [optimizer], [scheduler]

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
                patience=self.patience,
                mode="max",
                min_delta=0.00,
            )
            callbacks.append(early_stop)
        return callbacks
