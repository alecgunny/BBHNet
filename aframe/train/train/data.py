import glob
import logging
import os
from typing import Optional, Sequence

import h5py
import lightning.pytorch as pl
import torch
from train import augmentations as aug

from aframe.architectures.preprocessing import PsdEstimator
from ml4gw.dataloading import ChunkedDataset
from ml4gw.distributions import PowerLaw
from ml4gw.transforms import Whiten
from ml4gw.utils.slicing import sample_kernels, unfold_windows


class TimeSlideDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        timeseries: torch.Tensor,
        sample_rate: float,
        kernel_length: float,
        stride: float,
        batch_size: int,
        livetime: float,
        shift: float,
    ) -> None:
        super().__init__()
        self.timeseries = timeseries
        self.sample_rate = sample_rate
        self.kernel_size = int(kernel_length * sample_rate)
        self.stride_size = int(stride * sample_rate)
        self.batch_size = batch_size
        self.shift_size = int(shift * sample_rate)
        self.livetime_size = int(livetime * sample_rate)

    def steps_for_shift(self, shift: int):
        """Compute the number of kernels that will be taken per shift"""
        num_channels, size = self.timeseries.shape
        shift = abs(shift)  # doesn't matter which direction
        max_shift = shift * (num_channels - 1)

        size -= max_shift + self.kernel_size
        return size // self.stride_size + 1

    def get_step_size(self, batch_size):
        return self.stride_size * (batch_size - 1) + self.kernel_size

    def iter_timeslides(self):
        num_channels = len(self.timeseries)
        T = 0
        i = 1
        while True:
            shift = i * self.shift_size
            num_steps = self.steps_for_shift(shift)
            num_batches, remainder = divmod(num_steps, self.batch_size)
            if remainder:
                num_batches += 1

            shift_idx = [i * abs(shift) for i in range(num_channels)]
            if shift < 0:
                shift_idx.reverse()

            for j in range(num_batches):
                if (j + 1) == num_batches:
                    step_size = self.get_step_size(remainder)
                else:
                    step_size = self.get_step_size(self.batch_size)

                start = j * self.batch_size * self.stride_size
                background = []
                for k, offset in enumerate(shift_idx):
                    offset = start + offset
                    x = self.timeseries[k, offset : offset + step_size]
                    background.append(x)
                yield torch.stack(background)

                T += self.stride_size * self.batch_size
                if T >= self.livetime_size:
                    break
            else:
                # The loop didn't break, so we properly
                # exhausted that shift and we're ready
                # to move on to the next. Do the positive
                # and negative shifts for each shift value
                i *= -1
                if i > 0:
                    i += 1
                continue
            break

    def __iter__(self):
        return self.iter_timeslides()


# TODO: using this right now because
# lightning.pytorch.utilities.CombinedLoader
# is not supported when calling `.fit`. Once
# this has been fixed in
# https://github.com/Lightning-AI/lightning/issues/16830,
# we should switch to using a CombinedLoader for validation
class ZippedDataset(torch.utils.data.IterableDataset):
    def __init__(self, *datasets):
        super().__init__()
        self.datasets = datasets

    def __iter__(self):
        return zip(*self.datasets)


class AframeDataset(pl.LightningDataModule):
    def __init__(
        self,
        # data loading args
        data_dir: str,
        ifos: Sequence[str],
        valid_frac: float,
        # preprocessing args
        batch_size: int,
        kernel_length: float,
        fduration: float,
        psd_length: float,
        # augmentation args
        waveform_prob: float,
        swap_frac: float,
        mute_frac: float,
        snr_thresh: float = 4,
        max_snr: float = 100,
        snr_alpha: float = 3,
        trigger_pad: float = 0,
        fftlength: Optional[float] = None,
        highpass: Optional[float] = None,
        # validation args
        valid_stride: Optional[float] = None,
        num_valid_views: int = 4,
        valid_livetime: float = (3600 * 12),
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.num_ifos = len(ifos)
        self._logger = logging.getLogger("AframeDataset")

        # infer the sample rate from the data
        with h5py.File(self.train_fnames[0], "r") as f:
            sample_rate = 1 / f[ifos[0]].attrs["dx"]
        self.sample_rate = sample_rate
        self._logger.info(f"Inferred sample rate {sample_rate}")

        # set up some of the modules we'll need for
        # 1. Preprocessing
        fftlength = fftlength or kernel_length + fduration
        self.psd_estimator = PsdEstimator(
            kernel_length + fduration,
            sample_rate,
            fftlength,
            fast=highpass is not None,
        )
        self.whitener = Whiten(fduration, sample_rate, highpass)

        # 2. Data augmentation
        waveform_prob /= 1 + swap_frac * mute_frac - swap_frac - mute_frac
        self.inverter = aug.SignalInverter(0.5)
        self.reverser = aug.SignalReverser(0.5)
        self.snr_sampler = PowerLaw(snr_thresh, max_snr, snr_alpha)
        self.projector = aug.WaveformProjector(ifos, sample_rate, highpass)
        self.swapper = aug.ChannelSwapper(swap_frac)
        self.muter = aug.ChannelMuter(mute_frac)
        self.waveform_sampler = None

    @property
    def sample_length(self) -> float:
        return (
            self.hparams.kernel_length
            + self.hparams.fduration
            + self.hparams.psd_length
        )

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
        if self.waveform_sampler is None:
            train_frac = 1 - self.hparams.valid_frac
            with h5py.File(f"{self.hparams.data_dir}/signals.h5", "r") as f:
                num_waveforms = int(len(f["signals"]) * train_frac)
        else:
            num_waveforms = self.waveform_sampler.num_waveforms
        total_batches = int(4 * num_waveforms / waveforms_per_batch)
        return total_batches

    @property
    def val_batch_size(self):
        return int(4 * self.hparams.batch_size / len(self.trainer.device_ids))

    @torch.no_grad()
    def project_val_waveforms(self, waveforms, dec, psi, phi, psd):
        device = psd.device
        num_batches = (len(waveforms) - 1) // self.val_batch_size + 1
        responses = []
        for i in range(num_batches):
            slc = slice(i * self.val_batch_size, (i + 1) * self.val_batch_size)
            params = [i[slc].to(device) for i in [dec, phi, psi]]
            response = self.projector(
                *params,
                snrs=self.hparams.snr_thresh,
                psds=psd,
                cross=waveforms[slc, 0].to(device),
                plus=waveforms[slc, 1].to(device),
            )
            responses.append(response.cpu())
        return torch.cat(responses, dim=0)

    def get_device(self):
        if len(self.trainer.device_ids) > 1:
            rank = int(os.environ["LOCAL_RANK"])
            device_id = self.trainer.device_ids[rank]
            return f"cuda:{device_id}"
        elif isinstance(
            self.trainer.accelerator, pl.accelerators.CUDAAccelerator
        ):
            return f"cuda:{self.trainer.device_ids[0]}"
        else:
            return "cpu"

    def setup(self, stage: str):
        self._logger.info("Loading waveforms")
        with h5py.File(f"{self.hparams.data_dir}/signals.h5", "r") as f:
            signals = f["signals"][:]
            num_signals = len(signals)
            num_valid_signals = int(self.hparams.valid_frac * num_signals)

            self._logger.info(f"Loaded {num_signals} waveforms")
            self._logger.info(
                f"Reserving {num_valid_signals} waveforms for validation"
            )

            train_signals = torch.Tensor(signals[:-num_valid_signals])
            val_signals = torch.Tensor(signals[-num_valid_signals:])
            val_dec = torch.Tensor(f["dec"][-num_valid_signals:])
            val_psi = torch.Tensor(f["psi"][-num_valid_signals:])
            val_phi = torch.Tensor(f["ra"][-num_valid_signals:])

        self.waveform_sampler = aug.WaveformSampler(
            self.hparams.waveform_prob,
            cross=train_signals[:, 0],
            plus=train_signals[:, 1],
        )

        self._logger.info("Project validation waveforms to IFO responses")
        val_background = []
        with h5py.File(self.valid_fnames[0], "r") as f:
            for ifo in self.hparams.ifos:
                val_background.append(torch.Tensor(f[ifo][:]))
        self.val_background = torch.stack(val_background)

        device = self.get_device()
        psd = self.psd_estimator.spectral_density(self.val_background.double())
        psd = psd.to(device)
        self.val_waveforms = self.project_val_waveforms(
            val_signals, val_dec, val_psi, val_phi, self.val_background
        )

        # move all our modules with buffers to our local device
        self.projector.to(device)
        self.psd_estimator.to(device)
        self.whitener.to(device)

        self._logger.info("Initial dataloading complete")

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
            batch = self.build_val_batches(background, signals)
        return batch

    def augment(
        self, X: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        X, psds = self.psd_estimator(X)

        X = self.inverter(X)
        X = self.reverser(X)
        *params, polarizations, mask = self.waveform_sampler(X)

        N = len(params[0])
        snrs = self.snr_sampler(N).to(X.device)
        responses = self.projector(*params, snrs, psds[mask], **polarizations)
        kernels = sample_kernels(
            responses,
            kernel_size=X.size(-1),
            max_center_offset=self.pad_size,
            coincident=True,
        )

        # perform augmentations on the responses themselves,
        # keep track of which indices have been augmented
        kernels, swap_indices = self.swapper(kernels)
        kernels, mute_indices = self.muter(kernels)

        # inject the IFO responses
        X[mask] += kernels

        # mark which responses got augmented
        # so that we don't mark these as signal
        idx = torch.where(mask)[0]
        mask[idx[mute_indices]] = 0
        mask[idx[swap_indices]] = 0
        y[mask] += 1

        X = self.whitener(X, psds)
        return X, y

    def build_val_batches(self, background, signals):
        sample_size = int(self.sample_length * self.sample_rate)
        stride = int(self.hparams.valid_stride * self.sample_rate)
        background = unfold_windows(background, sample_size, stride=stride)

        X, psd = self.psd_estimator(background)
        X_bg = self.whitener(X, psd)

        # sometimes at the end of a segment,
        # there won't be enough background
        # kernels and so we'll have to inject
        # our signals on overlapping data and
        # ditch some at the end
        step = int(len(X) / len(signals))
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
        X_inj = torch.stack(X_inj)
        return X_bg, X_inj

    def val_dataloader(self) -> ZippedDataset:
        # how much larger do we want to make validation
        # batches than the batch size specified for
        # training
        background_dataset = TimeSlideDataset(
            self.val_background,
            self.sample_rate,
            self.sample_length,
            self.hparams.valid_stride,
            self.val_batch_size,
            self.hparams.valid_livetime,
            shift=1,
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
            / self.val_batch_size
        )
        num_waveforms = len(self.val_waveforms)
        signal_batch_size = (num_waveforms - 1) // num_val_batches + 1
        signal_dataset = torch.utils.data.TensorDataset(self.val_waveforms)
        signal_loader = torch.utils.data.DataLoader(
            signal_dataset,
            batch_size=signal_batch_size,
            shuffle=False,
            pin_memory=False,
        )
        return ZippedDataset(background_dataset, signal_loader)

    def train_dataloader(self) -> ChunkedDataset:
        chunks_per_epoch = 80
        reads_per_chunk = 10
        num_workers = 1

        batches_per_chunk = int(self.steps_per_epoch / chunks_per_epoch)
        device = self.get_device()
        pin_memory = "cuda" in device
        batch_per_device = int(
            self.hparams.batch_size / len(self.trainer.device_ids)
        )
        return ChunkedDataset(
            self.train_fnames,
            channels=self.hparams.ifos,
            kernel_length=self.sample_length,
            sample_rate=self.sample_rate,
            batch_size=batch_per_device,
            reads_per_chunk=reads_per_chunk,
            chunk_length=1024,
            batches_per_chunk=batches_per_chunk,
            chunks_per_epoch=chunks_per_epoch,
            num_workers=num_workers,
            device=device,
            pin_memory=pin_memory,
        )
