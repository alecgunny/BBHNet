from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
from scipy.signal import convolve

from bbhnet.io.h5 import write_timeseries
from bbhnet.io.timeslides import Segment


def boxcar_filter(y, window_size: int):
    window = np.ones((window_size,)) / window_size
    mf = convolve(y, window, mode="full")
    return mf[: -window_size + 1]


def analyze_segment(
    segment: Segment,
    window_length: float = 1,
    kernel_length: float = 1,
    norm_seconds: Optional[float] = None,
    write_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analyze a segment of time-contiguous BBHNet outputs

    Compute matched filter outputs on a stretch
    of frame files that are assumed to be contiguous
    and ordered in time. Matched filters are computed
    as the average over the the last `window_length`
    seconds of data, optionally normalized by the mean
    and standard deviation of the previous `norm_seconds`
    seconds worth of data.

    Args:
        segment:
            Segment of contiguous HDF5 files to analyze
        window_length:
            The length of time, in seconds, over which previous
            network outputs should be averaged to produce
            "matched filter" outputs.
        kernel_length:
            The length of time, in seconds, of the input kernel
            to BBHNet used to produce the outputs being analyzed
        norm_seconds:
            The number of seconds before each matched filter
            window over which to compute the mean and
            standard deviation of network outputs, which will
            be used to normalize filter outputs. If left as
            `None`, filter outputs won't be normalized.
        write_dir:
            A directory to write outputs to rather than return
            them. If left as `None`, the output arrays are returned
    Returns:
        Array of timestamps corresponding to the
            _end_ of the input kernel that would produce
            the corresponding network output and matched
            filter output
        Array of raw neural network outputs for each timestamp
        Array of matched filter outputs for each timestamp
    """

    # if we specified a normalization period, ensure
    # that we have at least 50% of that period to analyze
    if norm_seconds is not None and segment.length < (1.5 * norm_seconds):
        raise ValueError(
            "Segment {} has length {}s, but analysis "
            "requires at least {}s of data".format(
                segment, segment.length, 1.5 * norm_seconds
            )
        )

    # read in all the data for a given segment
    y, t = segment.load("out")
    sample_rate = 1 / (t[1] - t[0])
    t += kernel_length
    mf = boxcar_filter(y, window_size=int(window_length * sample_rate))

    if norm_seconds is not None:
        # compute the mean and standard deviation of
        # the last `norm_seconds` seconds of data
        # compute the standard deviation by the
        # sigma^2 = E[x^2] - E^2[x] trick
        shifts = boxcar_filter(y, int(norm_seconds * sample_rate))
        sqs = boxcar_filter(y**2, int(norm_seconds * sample_rate))
        scales = np.sqrt(sqs - shifts**2)

        # get rid of the first norm_seconds worth of data
        # since there's nothing to normalize by
        idx = int(norm_seconds * sample_rate)
        mf = (mf[idx:] - shifts[idx:]) / scales[idx:]
        t = t[idx:]
        y = y[idx:]

    # advance timesteps by the kernel length
    # so that the represent the last sample
    # of a kernel rather than the first
    if write_dir is not None:
        shift = Path(segment.fnames[0]).parts[-4]
        write_dir = write_dir / shift
        write_dir.mkdir(parents=True, exist_ok=True)
        fname = write_timeseries(write_dir, t=t, y=y, filtered=mf)
        return fname, mf.min(), mf.max()
    return t, y, mf


def analyze_segments_parallel(
    segments: Iterable[Segment],
    window_length: float = 1,
    kernel_length: float = 1,
    norm_seconds: Optional[float] = None,
    write_dir: Optional[Path] = None,
    num_proc: int = 1,
):
    futures = []
    ex = ProcessPoolExecutor(num_proc)
    for segment in segments:
        future = ex.submit(
            analyze_segment,
            segment,
            window_length=window_length,
            kernel_length=kernel_length,
            norm_seconds=norm_seconds,
            write_dir=write_dir,
        )
        futures.append(future)

    try:
        for future in as_completed(futures):
            exc = future.exception()
            if exc is not None:
                raise exc

            yield future.result()
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
