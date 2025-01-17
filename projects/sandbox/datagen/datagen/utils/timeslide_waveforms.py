import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from gwpy.timeseries import TimeSeriesDict

from aframe.utils.timeslides import calc_shifts_required


def get_num_shifts(
    segments: List[Tuple[float, float]], Tb: float, shift: float
) -> int:
    """
    Calculates the number of required time shifts based on a list
    of background segments
    """
    T = sum([stop - start for start, stop in segments])
    return calc_shifts_required(Tb, T, shift)


def io_with_blocking(f, fname, timeout=10):
    """
    Function that assists with multiple processes writing to the same file
    """
    start_time = time.time()
    while True:
        try:
            return f(fname)
        except BlockingIOError:
            if (time.time() - start_time) > timeout:
                raise


def load_psds(background: Path, ifos: List[str], df: float) -> torch.Tensor:
    """Calculate PSDs from background generated by `background.py`"""
    background = TimeSeriesDict.read(background, path=ifos)
    psds = []
    for ifo in ifos:
        psd = background[ifo].psd(1 / df, window="hann", method="median")
        psds.append(psd.value)
    psds = torch.tensor(np.stack(psds), dtype=torch.float64)
    return psds


def calc_segment_injection_times(
    start: float,
    stop: float,
    spacing: float,
    buffer: float,
    waveform_duration: float,
):
    """
    Calculate the times at which to inject signals into a segment

    Args:
        start:
            The start time of the segment
        stop:
            The stop time of the segment
        spacing:
            The amount of time, in seconds, to leave between the end
            of one signal and the start of the next
        buffer:
            The amount of time, in seconds, on either side of the
            segment within which injection times will not be
            generated
        waveform_duration:
            The duration of the waveform in seconds

    Returns: np.ndarray of injection times
    """

    buffer += waveform_duration // 2
    spacing += waveform_duration
    injection_times = np.arange(start + buffer, stop - buffer, spacing)
    return injection_times
