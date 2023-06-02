import logging
from math import pi
from pathlib import Path
from typing import List, Tuple, TypeVar

import h5py
import numpy as np
import torch
from train.augmentor import AframeBatchAugmentor
from train.data_structures import SnrRescaler, SnrSampler

import ml4gw.gw as gw
from ml4gw.distributions import Cosine, Uniform

Tensor = TypeVar("Tensor", np.ndarray, torch.Tensor)


def split(X: Tensor, frac: float, axis: int) -> Tuple[Tensor, Tensor]:
    """
    Split an array into two parts along the given axis
    by an amount specified by `frac`. Generic to both
    numpy arrays and torch Tensors.
    """

    size = int(frac * X.shape[axis])
    # Catches fp error that sometimes happens when size should be an exact int
    # Is there a better way to do this?
    if np.abs(frac * X.shape[axis] - size - 1) < 1e-10:
        size += 1

    if isinstance(X, np.ndarray):
        return np.split(X, [size], axis=axis)
    else:
        splits = [size, X.shape[axis] - size]
        return torch.split(X, splits, dim=axis)


def _sort_key(fname: Path):
    return int(fname.stem.split("-")[-2])


def get_background_fnames(data_dir: Path):
    fnames = data_dir.glob("*.hdf5")
    fnames = sorted(fnames, key=_sort_key)
    return list(fnames)


def get_background(fname: Path):
    background = []
    with h5py.File(fname, "r") as f:
        ifos = list(f.keys())
        for ifo in ifos:
            hoft = f[ifo][:]
            background.append(hoft)
    return np.stack(background)


def get_waveforms(
    waveform_dataset: Path,
    ifos: List[str],
    sample_rate: float,
    valid_frac: float,
):
    # perform train/val split of waveforms,
    # and compute fixed validation responses
    with h5py.File(waveform_dataset, "r") as f:
        signals = f["signals"][:]

        if valid_frac is not None:
            signals, valid_signals = split(signals, 1 - valid_frac, 0)

            valid_cross, valid_plus = valid_signals.transpose(1, 0, 2)
            slc = slice(-len(valid_signals), None)
            dec, psi, phi = f["dec"][slc], f["psi"][slc], f["ra"][slc]

            # project the validation waveforms to IFO
            # responses up front since we don't care
            # about sampling sky parameters
            tensors, vertices = gw.get_ifo_geometry(*ifos)
            valid_responses = gw.compute_observed_strain(
                torch.Tensor(dec),
                torch.Tensor(psi),
                torch.Tensor(phi),
                detector_tensors=tensors,
                detector_vertices=vertices,
                sample_rate=sample_rate,
                plus=torch.Tensor(valid_plus),
                cross=torch.Tensor(valid_cross),
            )
            return signals, valid_responses
    return signals, None


def threshold_snrs(
    ifo_responses: torch.Tensor,
    threshold: float,
    sample_rate: float,
    psd: torch.Tensor,
    highpass: float,
):
    mask = torch.linspace(0, sample_rate / 2, psd.shape[-1])
    mask = mask >= highpass
    mask = mask.to(ifo_responses.device)

    snrs = gw.compute_network_snr(ifo_responses, psd, sample_rate, mask)
    target_snrs = snrs.clamp(threshold, 1000)
    weights = target_snrs / snrs

    num_rescaled = (weights > 1).sum().item()
    logging.info(
        "{}/{} waveforms had SNR<{}".format(
            num_rescaled, len(weights), threshold
        )
    )

    return ifo_responses * weights.view(-1, 1, 1)


def get_augmentor(
    ifos: List[str],
    sample_rate: float,
    waveforms: np.ndarray,
    waveform_prob: float,
    snr_sampler: SnrSampler,
    snr_rescaler: SnrRescaler,
    mute_frac: float,
    swap_frac: float,
    trigger_distance: float,
    invert_prob: float = 0.5,
    reverse_prob: float = 0.5,
):
    cross, plus = waveforms.transpose(1, 0, 2)
    return AframeBatchAugmentor(
        ifos,
        sample_rate,
        waveform_prob,
        dec=Cosine(),
        psi=Uniform(0, pi),
        phi=Uniform(-pi, pi),
        trigger_distance=trigger_distance,
        mute_frac=mute_frac,
        swap_frac=swap_frac,
        snr=snr_sampler,
        rescaler=snr_rescaler,
        invert_prob=invert_prob,
        reverse_prob=reverse_prob,
        cross=cross,
        plus=plus,
    )
