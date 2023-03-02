import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np

from bbhnet.analysis.events import LigoResponseSet


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
            "background": np.zeros((num_predictions,)),
            "foreground": np.zeros((num_predictions,)),
        }
        self._done = {self.sequence_id + i: False for i in range(2)}

        self.num_steps = num_steps
        self.initialized = False

        self.injection_set = LigoResponseSet.read(
            self.injection_set_file, start=self.start, end=self.stop
        )

    @property
    def done(self):
        return all(self._done.values())

    @property
    def pad(self):
        return self.injection_set.waveform_duration / 2

    def update(self, y: np.ndarray, request_id: int, sequence_id: int) -> bool:
        self.initialized = True

        y = y[:, 0]
        start = request_id * self.batch_size
        stop = (request_id + 1) * self.batch_size
        self.predictions[sequence_id][start:stop] = y
        done = (request_id + 1) == self.num_steps
        self._done[sequence_id] = done
        return done

    def iter(
        self,
        data_it: Iterator,
        sample_rate: float,
        throughput: float,
    ) -> Tuple[int, np.ndarray]:
        """
        Generate streaming snapshot state updates
        from a chunked dataloader at the specified
        throughput.
        """
        window_stride = int(sample_rate / self.sample_rate)
        step_size = self.batch_size * window_stride

        inf_per_second = throughput * self.sample_rate
        batches_per_second = inf_per_second / self.batch_size
        sleep = 1 / batches_per_second

        # grab data up front and refresh it when we need it
        x = next(data_it).astype("float32")
        inj_x = self.injection_set.inject(x.copy(), self.start)

        global_idx, chunk_idx = 0, 0
        while global_idx < self.num_steps:
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
                    x = next(data_it).astype("float32")
                except StopIteration:
                    raise ValueError(
                        "Ran out of data at iteration {} when "
                        "{} iterations were expected".format(
                            global_idx, self.num_steps
                        )
                    )

                # prepend any data leftover from the last chunk
                if remainder is not None:
                    x = np.concatenate([remainder, x], axis=1)

                # inject on the newly loaded data
                start = self.start + global_idx * step_size / sample_rate
                inj_x = self.injection_set.inject(x.copy(), start)

                # reset our per-chunk counters
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
