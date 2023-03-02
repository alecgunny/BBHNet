import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
from ratelimiter import RateLimiter

from bbhnet.analysis.ledger.injections import LigoResponseSet


def _intify(x):
    return int(x) if int(x) == x else x


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
        if remainder is not None:
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
    return x, new_remainders


@dataclass
class SegmentIterator:
    it: Iterator
    start: float
    end: float
    sample_rate: float
    shifts: List[float]

    @property
    def duration(self):
        return self.end - self.start

    def __iter__(self):
        return self._shift_it()

    def _shift_it(self):
        shifts = [int(i * self.sample_rate) for i in self.shifts]
        remainders = [None for _ in shifts]
        for x in self.it:
            x, remainders = _shift_chunk(x, shifts, remainders)
            yield x

    def __str__(self) -> str:
        return f"{_intify(self.start)}-{_intify(self.end)}"


@dataclass
class Subsequence:
    size: int

    def __post_init__(self):
        self.y = np.zeros((self.size,))
        self._idx = 0

    @property
    def done(self):
        return self._idx >= self.size

    @property
    def initialized(self):
        return self._idx > 0

    def update(self, y):
        self.y[self._idx : self._idx + len(y)] = y[:, 0]
        self._idx += len(y)


@dataclass
class Sequence:
    segment: SegmentIterator
    injection_set_file: Path
    sample_rate: float
    batch_size: int
    throughput: float

    def __post_init__(self):
        num_predictions = self.segment.duration * self.sample_rate
        num_steps = int(num_predictions // self.batch_size)
        num_predictions = int(num_steps * self.batch_size)

        self.num_steps = num_steps
        self.background = Subsequence(num_predictions)
        self.foreground = Subsequence(num_predictions)
        self.injection_set = LigoResponseSet.read(
            self.injection_set_file,
            start=self.segment.start,
            end=self.segment.end,
        )

    @property
    def done(self):
        return self.background.done & self.foreground.done

    @property
    def initialized(self):
        return self.background.initialized & self.foreground.initialized

    def __len__(self):
        return self.num_steps

    def __iter__(self):
        return self.iter_updates()

    def inject(self, x, i):
        start = self.segment.start + i * self.batch_size / self.sample_rate
        return self.injection_set.inject(x.copy(), start)

    def iter_updates(self):
        """
        Generate streaming snapshot state updates
        from a chunked dataloader at the specified
        throughput.
        """
        window_stride = int(self.segment.sample_rate / self.sample_rate)
        step_size = self.batch_size * window_stride

        inf_per_second = self.throughput * self.sample_rate
        batches_per_second = inf_per_second / self.batch_size

        max_calls = 2
        period = 1.5 * max_calls / batches_per_second
        rate_limiter = RateLimiter(max_calls=max_calls, period=period)

        # grab data up front and refresh it when we need it
        it = iter(self.segment)
        x = next(it).astype("float32")
        inj_x = self.inject(x, 0)

        chunk_idx = 0
        for i in range(len(self)):
            start = chunk_idx * step_size
            stop = (chunk_idx + 1) * step_size

            # if we can't build an entire batch with
            # whatever data we have left, grab the
            # next chunk of data
            if stop > x.shape[-1]:
                # check if there will be any data
                # leftover at the end of this chunk
                if start < x.shape[-1]:
                    remainder = x[:, start:]
                else:
                    remainder = None

                try:
                    x = next(it).astype("float32")
                except StopIteration:
                    raise ValueError(
                        "Ran out of data at iteration {} when {} "
                        "iterations were expected".format(i, len(self))
                    )

                # prepend any data leftover from the last chunk
                if remainder is not None:
                    x = np.concatenate([remainder, x], axis=1)

                # inject on the newly loaded data
                inj_x = self.inject(x, i)

                # reset our per-chunk counters
                chunk_idx = 0
                start, stop = 0, step_size

            with rate_limiter:
                yield x[:, start:stop], inj_x[:, start:stop]

            chunk_idx += 1
            while not self.initialized:
                time.sleep(1e-3)

    def __str__(self):
        return str(self.segment)


def get_segments(data_dir: Path) -> List[Tuple[float, float]]:
    return []


def load_sequences(
    data_dir: Path,
    ifos: List[str],
    chunk_size: float,
    shifts: List[float],
    injection_set_file: float,
    sample_rate: float,
    inference_sampling_rate: float,
    batch_size: int,
    throughput: float,
) -> Tuple[Tuple[float, float], Sequence]:
    from mldatafind.find import find_data

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
    for (start, end), it in zip(segments, data_it):
        segment = SegmentIterator(it, start, end, sample_rate, shifts)
        sequence = Sequence(
            segment,
            injection_set_file,
            inference_sampling_rate,
            batch_size,
            throughput,
        )
        yield (start, end), sequence
