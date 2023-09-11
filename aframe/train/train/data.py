import glob
import itertools
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

import h5py
import lightning.pytorch as pl
import torch
from lightning.fabric import Fabric
from train import augmentations as aug

from aframe.architectures.preprocessing import PsdEstimator
from ml4gw.dataloading import Hdf5TimeSeriesDataset
from ml4gw.distributions import PowerLaw
from ml4gw.transforms import Whiten
from ml4gw.utils.slicing import sample_kernels, unfold_windows


def x_per_y(x, y):
    return int((x - 1) // y) + 1


@dataclass
class TimeSlide:
    timeseries: torch.Tensor
    shift_size: int
    kernel_size: float
    stride_size: float
    batch_size: int

    def __post_init__(self):
        num_channels, size = self.timeseries.shape
        max_shift = abs(self.shift_size) * (num_channels - 1)

        size -= max_shift + self.kernel_size
        num_steps = size // self.stride_size + 1
        self.num_batches, self.remainder = divmod(num_steps, self.batch_size)

        shift_idx = [i * abs(self.shift_size) for i in range(num_channels)]
        if self.shift_size < 0:
            shift_idx.reverse()
        self.shift_idx = shift_idx

    def __len__(self):
        return self.num_batches + int(self.remainder > 0)

    def steps_for_batch(self, batch_size=None):
        batch_size = batch_size or self.batch_size
        return self.stride_size * (batch_size - 1) + self.kernel_size

    def get_batch(self, start, batch_size=None):
        step_size = self.steps_for_batch(batch_size)
        background = []
        for j, offset in enumerate(self.shift_idx):
            offset = start + offset
            x = self.timeseries[j, offset : offset + step_size]
            background.append(x)
        return torch.stack(background)

    def __iter__(self):
        for i in range(self.num_batches):
            start = i * self.batch_size * self.stride_size
            yield self.get_batch(start)

        if self.remainder:
            start = (i + 1) * self.batch_size * self.stride_size
            yield self.get_batch(start, self.remainder)


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

    def __len__(self):
        lengths = []
        for dset in self.datasets:
            try:
                lengths.append(len(dset))
            except Exception as e:
                raise e from None
        return min(lengths)

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
            average="median",
        )
        self.whitener = Whiten(fduration, sample_rate, highpass)

        # 2. Data augmentation
        self.inverter = aug.SignalInverter(0.5)
        self.reverser = aug.SignalReverser(0.5)
        self.snr_sampler = PowerLaw(snr_thresh, max_snr, snr_alpha)
        self.projector = aug.WaveformProjector(ifos, sample_rate, highpass)
        self.swapper = aug.ChannelSwapper(swap_frac)
        self.muter = aug.ChannelMuter(mute_frac)
        self.waveform_sampler = None

    @property
    def sample_length(self) -> float:
        """Length of samples generated by datasets in seconds"""
        return (
            self.hparams.kernel_length
            + self.hparams.fduration
            + self.hparams.psd_length
        )

    @property
    def pad_size(self) -> int:
        """
        Number of samples away from edge of kernel to ensure
        that waveforms are injected at.
        """
        return int(self.hparams.trigger_pad * self.sample_rate)

    # TODO: should we come up with a more clever scheme for
    # breaking up our training and validation background data?
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
        """
        Number of gradient updates between validation periods,
        taking into account number of devices currently being
        utilized for training implicitly through the number
        of waveforms on the local device.
        """

        waveforms_per_batch = (
            self.hparams.batch_size * self.hparams.waveform_prob
        )

        # we don't load in waveforms until setup() gets called,
        # so in case we need this between init time and then,
        # grab the number of training waveforms explicitly.
        if self.waveform_sampler is None:
            train_frac = 1 - self.hparams.valid_frac
            with h5py.File(f"{self.hparams.data_dir}/signals.h5", "r") as f:
                num_waveforms = int(len(f["signals"]) * train_frac)
        else:
            # otherwise it should be saved as an attribute here
            num_waveforms = self.waveform_sampler.num_waveforms

        # multiply by 4 to account for the fact that
        # sky parameter sampling means we won't see
        # the same waveform the same way, and so we
        # technically have "more" data.
        total_batches = int(4 * num_waveforms / waveforms_per_batch)
        return total_batches

    @property
    def val_batch_size(self):
        """Use larger batch sizes when we don't need gradients."""
        return int(4 * self.hparams.batch_size)

    @torch.no_grad()
    def project_val_waveforms(self, waveforms, dec, psi, phi, psd):
        """
        Pre-project validation waveforms to interferometer
        responses and threshold their SNRs at our minimum value.
        """

        device = psd.device
        num_batches = x_per_y(len(waveforms), self.val_batch_size)
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

    def build_background_dataset(self, timeseries):
        duration = timeseries.size(-1) / self.sample_rate
        duration -= self.hparams.psd_length

        kernel_size = int(self.hparams.kernel_length * self.sample_rate)
        stride_size = int(self.hparams.valid_stride * self.sample_rate)

        # create alternating positive and negative
        # shifts until we have enough livetime
        livetime = self.hparams.valid_livetime + 0
        shifts = zip(itertools.count(1, 1), itertools.count(-1, -1))
        shifts = itertools.chain.from_iterable(shifts)
        shifts = itertools.takewhile(lambda _: livetime > 0, shifts)
        timeslides = []
        for shift in shifts:
            timeslide = TimeSlide(
                timeseries,
                int(shift * self.sample_rate),
                kernel_size,
                stride_size,
                self.val_batch_size,
            )
            livetime -= duration - abs(shift)
            timeslides.append(timeslide)

        if Fabric.world_size > 1 and Fabric.global_rank > len(timeslides):
            self._logger.info(
                f"Skipping validation on rank {Fabric.global_rank}"
            )
            self.timeslides = []
            return None, None
        elif Fabric.world_size == 1:
            self.timeslides = timeslides
            return 0, 1

        total_shifts = len(timeslides)
        shifts_per_dev = x_per_y(total_shifts, Fabric.world_size)

        start = Fabric.global_rank * shifts_per_dev
        stop = (Fabric.global_rank + 1) * shifts_per_dev
        self.timeslides = timeslides[start:stop]

        shifts = [i.shift_size / self.sample_rate for i in self.timeslides]
        self._logger.info(
            "Validating shifts {} on rank {}".format(
                ",".join(map(str, shifts)), Fabric.global_rank
            )
        )
        return start / total_shifts, len(timeslides) / total_shifts

    def setup(self, stage: str):
        # move all our modules with buffers to our local device
        self.projector.to(Fabric.device)
        self.whitener.to(Fabric.device)

        # load in our validation background up front and
        # compute which timeslides we'll do on this device
        # if we're doing distributed training so we'll know
        # which waveforms to subsample
        self._logger.info("Loading validation background data")
        val_background = []
        with h5py.File(self.valid_fnames[0], "r") as f:
            for ifo in self.hparams.ifos:
                val_background.append(torch.Tensor(f[ifo][:]))
        val_background = torch.stack(val_background)
        psd = self.psd_estimator.spectral_density(val_background.double())

        # now move the PSD and the estimator onto the desired device
        psd = psd.to(Fabric.device)
        self.psd_estimator.to(Fabric.device)
        start, frac = self.build_background_dataset(val_background)

        self._logger.info("Loading waveforms")
        with h5py.File(f"{self.hparams.data_dir}/signals.h5", "r") as f:
            num_signals = len(f["signals"])
            num_valid_signals = int(self.hparams.valid_frac * num_signals)
            num_train_signals = num_signals - num_valid_signals
            if not Fabric.global_rank:
                self._logger.info(
                    "Training on {} waveforms, with {} "
                    "reserved for validation".format(
                        num_train_signals, num_valid_signals
                    )
                )

            if Fabric.world_size > 1:
                signals_per_dev = x_per_y(num_train_signals, Fabric.world_size)
                start = Fabric.global_rank * signals_per_dev
                stop = (Fabric.global_rank + 1) * signals_per_dev
            else:
                start, stop = 0, num_train_signals
            self._logger.info(
                "Loading {} train waveforms on rank {}".format(
                    stop - start, Fabric.global_rank
                )
            )
            train_signals = torch.Tensor(f["signals"][start:stop])
            self._logger.info("Training waveforms loaded")

            # increase the likilehood of sampling waveforms
            # by a factor that accounts for the fact that
            # not all of them will be marked as signal due
            # to swapping and muting
            upweight = (
                1
                + self.hparams.swap_frac * self.hparams.mute_frac
                - self.hparams.swap_frac
                - self.hparams.mute_frac
            )
            self.waveform_sampler = aug.WaveformSampler(
                self.hparams.waveform_prob / upweight,
                cross=train_signals[:, 0],
                plus=train_signals[:, 1],
            )
            if start is None:
                self.val_waveforms = None
                return

            # subsample which waveforms we're using
            # based on the fraction of shifts that
            # are getting done on this device
            device_signals = int(frac * num_valid_signals)
            start = -int(num_valid_signals * (1 - start))
            stop = (device_signals + start) or None
            slc = slice(start, stop)

            self._logger.info(
                "Loading {} validation waveforms on rank {}".format(
                    device_signals, Fabric.global_rank
                )
            )

            # grab the appropriate slice of validation waveforms
            val_signals = torch.Tensor(f["signals"][slc])
            val_dec = torch.Tensor(f["dec"][slc])
            val_psi = torch.Tensor(f["psi"][slc])
            val_phi = torch.Tensor(f["ra"][slc])

        self._logger.info("Projecting validation waveforms to IFO responses")
        # now finally project our raw polarizations into
        # inteferometer responses on this device using
        # the PSD from the entire background segment
        self.val_waveforms = self.project_val_waveforms(
            val_signals, val_dec, val_psi, val_phi, psd
        )
        self._logger.info("Initial dataloading complete")

    def on_after_batch_transfer(self, batch, _):
        if self.trainer.training:
            # if we're training, perform random augmentations
            # on input data and use it to impact labels
            [X] = batch
            batch = self.augment(X)
        elif self.trainer.validating or self.trainer.sanity_checking:
            # If we're in validation mode but we're not validating
            # on the local device, the relevant tensors will be
            # empty, so just pass them through with a 0 shift to
            # indicate that this should be ignored
            [background, _, timeslide_idx], [signals] = batch
            if not background.size(0):
                return 0, background, signals

            # If we're validating, unfold the background
            # data into a batch of overlapping kernels now that
            # we're on the GPU so that we're not transferring as
            # much data from CPU to GPU. Once everything is
            # on-device, pre-inject signals into background.
            shift = self.timeslides[timeslide_idx].shift_size
            background, signals = self.build_val_batches(background, signals)
            batch = (shift, background, signals)
        return batch

    def augment(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform augmentations on a training batch
        and generate labels indicating which samples
        have had injections performed on them.
        """

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

        # inject the IFO responses and whiten
        X[mask] += kernels
        X = self.whitener(X, psds)

        # mark which responses got augmented
        # so that we don't mark these as signal
        idx = torch.where(mask)[0]
        mask[idx[mute_indices]] = 0
        mask[idx[swap_indices]] = 0

        # make labels
        y = torch.zeros((X.size(0), 1), device=X.device)
        y[mask] += 1
        return X, y

    def build_val_batches(
        self, background: torch.Tensor, signals: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Unfold a timeseries of background data
        into a batch of kernels, then inject
        multiple views of the provided signals
        into these timeseries. Whiten all tensors
        and return both the background and injected
        batches.
        """

        # TODO: in the same way we do inference, should we
        # use a longer PSD length and do the whitening
        # before we do the windowing to reduce compute?
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
        if self.val_waveforms is None or not self.timeslides:
            empty = torch.Tensor([])
            it = [[(empty, 0, 0), [empty]]]
            return it

        background_dataset = pl.CombinedLoader(
            self.timeslides, mode="sequential"
        )

        # Figure out how many batches of background
        # we're going to go through, then batch the
        # signals so that they're spaced evenly
        # throughout all those batches.
        num_waveforms = len(self.val_waveforms)
        signal_batch_size = (num_waveforms - 1) // len(background_dataset) + 1
        signal_dataset = torch.utils.data.TensorDataset(self.val_waveforms)
        signal_loader = torch.utils.data.DataLoader(
            signal_dataset,
            batch_size=signal_batch_size,
            shuffle=False,
            pin_memory=False,
        )
        return ZippedDataset(background_dataset, signal_loader)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        pin_memory = isinstance(Fabric.device, pl.accelerators.CUDAAccelerator)
        num_workers = min(6, int(os.cpu_count() / Fabric.world_size))

        dataset = Hdf5TimeSeriesDataset(
            self.train_fnames,
            channels=self.hparams.ifos,
            kernel_size=int(self.sample_rate * self.sample_length),
            batch_size=self.hparams.batch_size,
            batches_per_epoch=self.steps_per_epoch,
            coincident=False,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            pin_memory=pin_memory,
            pin_memory_device=Fabric.device if pin_memory else None,
        )
        return dataloader
