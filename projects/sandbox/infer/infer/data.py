from itertools import repeat
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
from mldatafind.find import find_data


def get_segments(data_dir: Path) -> List[Tuple[float, float]]:
    return []


def _shift_chunk(
    x: np.ndarray, shifts: List[int], remainders: List[Optional[np.ndarray]]
):
    max_shift = max(shifts)
    channels, new_remainders = [], []
    for channel, shift, remainder in zip(x, shifts, remainders):
        # if channel has pre-existing data, then prepend
        # it and start at the beginning of pre-existing
        # data. Otherwise, we'll assume this is the first
        # chunk in a segment and shift away the data at
        # the start
        if channel is not None:
            channel = np.concatenate([remainder, channel])
            start = 0
        else:
            start = shift

        stop = shift - max_shift or None
        shifted = channel[start:stop]
        channels.append(shifted)
        if stop is not None:
            new_remainders.append(channel[stop:])
        else:
            new_remainders.append([])

    x = np.stack(channels)
    return channels, remainders


def shift_it(data_it: Iterator, shifts: List[float], sample_rate: float):
    shifts = [int(i * sample_rate) for i in shifts]
    remainders = [None for _ in shifts]
    for x in data_it:
        x, remainders = _shift_chunk(x, shifts, remainders)
        yield x


def load_segments(
    data_dir: Path,
    ifos: List[str],
    chunk_size: float,
    shifts: List[float],
    sample_rate: float,
) -> Tuple[Tuple[float, float], np.ndarray]:
    segments = get_segments(data_dir)
    data_it = find_data(
        segments,
        channels=ifos,
        chunk_size=chunk_size,
        data_dir=data_dir,
        array_like=True,
        n_workers=1,
        thread=False,
    )
    for segment, it in zip(segments, data_it):
        it = shift_it(it, shifts, sample_rate)
        segment_it = repeat(segment)
        yield zip(segment_it, it)
