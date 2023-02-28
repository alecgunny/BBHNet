import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np

from bbhnet.analysis import InjectionSet


def _intify(x):
    return int(x) if int(x) == x else x


@dataclass
class Sequence:
    start: float
    stop: float
    sample_rate: float
    batch_size: int
    sequence_id: int
    injection_set_file: Path

    def __post_init__(self):
        num_predictions = (self.stop - self.start) * self.sample_rate
        num_steps = int(num_predictions // self.batch_size)
        num_predictions = int(num_steps * self.batch_size)

        self.predictions = {
            self.sequence_id: np.zeros((num_predictions,)),
            self.sequence_id + 1: np.zeros((num_predictions,)),
        }

        self.num_steps = num_steps
        self.initialized = False

        self.injection_set = InjectionSet.read(
            self.injection_set_file, start=self.start, stop=self.stop
        )

    @property
    def pad(self):
        return self.injection_set.waveform_duration / 2

    def update(self, y: np.ndarray, request_id: int, sequence_id: int) -> bool:
        self.initialized = True

        y = y[:, 0]
        start = request_id * self.batch_size
        stop = (request_id + 1) * self.batch_size
        self.predictions[sequence_id][start:stop] = y
        return (request_id + 1) == self.num_steps

    def inject_ifo(
        self, x: np.ndarray, ifo: str, idx: int, sample_rate: float
    ):
        start = self.start + idx / self.sample_rate
        stop = start + x.shape[-1] / sample_rate

        mask = self.injection_set.times >= (start - self.pad)
        mask &= self.injection_set.times <= (stop + self.pad)
        times = self.injection_set.times[mask]
        waveforms = getattr(self.injection_set, ifo)[mask]

        # potentially pad x to inject waveforms
        # that fall over the boundaries of chunks
        pad = []
        early = (times - self.pad) < start
        if early.any():
            pad.append(early.sum())
        else:
            pad.append(0)

        late = (times + self.pad) > stop
        if late.any():
            pad.append(late.sum())
        else:
            pad.append(0)

        if any(pad):
            x = np.pad(x, pad)
        times = times - times[0]

        # create matrix of indices of waveform_size for each waveform
        waveform_size = waveforms.shape[-1]
        idx = np.arange(waveform_size)[None] - int(waveform_size // 2)
        idx = np.repeat(idx, len(waveforms), axis=0)

        # offset the indices of each waveform
        # according to their time offset
        idx_diffs = (times * sample_rate).astype("int64")
        idx += idx_diffs[:, None]

        # flatten these indices and the signals out
        # to 1D and then add them in-place all at once
        idx = idx.reshape(-1)
        waveforms = waveforms.reshape(-1)
        x[idx] += waveforms
        if any(pad):
            start, stop = pad
            stop = -stop or None
            x = x[start:stop]
        return x

    def inject(
        self, x: np.ndarray, ifos: List[str], idx: int, sample_rate: float
    ) -> np.ndarray:
        inj_x = np.zeros_like(x)
        for i, ifo in enumerate(ifos):
            inj = self.inject_ifo(x[i], ifo, idx, sample_rate)
            inj_x[i] = inj
        return inj_x

    def iter(
        self,
        data_it: Iterator,
        ifos: List[str],
        sample_rate: float,
        throughput: float,
    ) -> Tuple[int, np.ndarray]:
        """
        Generate streaming snapshot state updates
        from a chunked dataloader at the specified
        throughput.
        """

        step_size = self.batch_size * int(sample_rate / self.sample_rate)
        sleep = self.batch_size / self.sample_rate / throughput

        x = next(data_it).astype("float32")
        inj_x = self.inject(x, ifos, 0, sample_rate)

        global_idx, chunk_idx = 0, 0
        while global_idx < self.num_steps:
            start = chunk_idx * step_size
            stop = (chunk_idx + 1) * step_size

            if stop > x.shape[-1]:
                try:
                    x = next(data_it).astype("float32")
                except StopIteration:
                    raise ValueError(
                        "Ran out of data at iteration {} when "
                        "{} iterations were expected".format(
                            global_idx, self.num_steps
                        )
                    )

                idx = global_idx * step_size
                inj_x = self.inject(x, ifos, idx, sample_rate)
                chunk_idx = 0
                start, stop = 0, step_size

            yield x[:, start:stop], inj_x[:, start:stop]
            global_idx += 1
            chunk_idx += 1

            while not self.initialized:
                time.sleep(1e-3)
            time.sleep(sleep)

    def __str__(self) -> str:
        return f"{_intify(self.start)}-{_intify(self.stop)}"
