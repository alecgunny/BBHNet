import logging
import re
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from gwpy.timeseries import TimeSeries, TimeSeriesDict
from scipy.signal import convolve
from typeo import scriptify

from bbhnet.io.h5 import read_timeseries
from bbhnet.logging import configure_logging
from hermes.aeriel.client import InferenceClient


class SequenceNotStarted(Exception):
    pass


class Callback:
    def __init__(
        self,
        write_dir: Path,
        inference_sampling_rate: float,
        batch_size: int,
        window_length: float,
    ) -> None:
        self.write_dir = write_dir
        self.inference_sampling_rate = inference_sampling_rate
        self.batch_size = batch_size
        self.window_size = int(inference_sampling_rate * window_length)
        self.window = np.ones((self.window_size,)) / self.window_size

    def start_new_sequence(self, start, stop):
        num_predictions = (stop - start) * self.inference_sampling_rate
        num_steps = int(num_predictions // self.batch_size)
        num_predictions = int(num_steps * self.batch_size)
        self._predictions = np.zeros((num_predictions,))

        self._num_steps = num_steps
        self._start = start
        self._stop = stop
        self.initialized = False
        self._past = np.zeros((0,))
        return num_steps

    def _make_timeseries(self, x, t0):
        return TimeSeries(x, sample_rate=self.inference_sampling_rate, t0=t0)

    def flush(self):
        predictions = self._make_timeseries(self._predictions, self._start)

        integrated = convolve(self._predictions, self.window, mode="full")
        integrated = integrated[: -self.window_size + 1]
        offset = self.window_size / self.inference_sampling_rate
        integrated = self._make_timeseries(integrated, self._start + offset)

        total_inferences = self._num_steps * self.batch_size
        duration = self.inference_sampling_rate * total_inferences
        fname = self.write_dir / f"out_{self._start}-{duration}.hdf5"

        tsdict = TimeSeriesDict({"raw": predictions, "integrated": integrated})
        tsdict.write(fname)
        logging.info(f"Wrote data to file {fname}")

        self._start = self._stop = None
        return fname

    def __call__(self, y, request_id, sequence_id):
        if self._start is None:
            raise SequenceNotStarted
        self.initialized = True

        y = y[:, 0]
        start = request_id * self.batch_size
        stop = request_id * self.batch_size
        self._predictions[start:stop] = y

        integrated = self.integrate(y)
        self._integrated[start:stop] = integrated

        if (request_id + 1) == self._num_steps:
            return self.flush()


def shift_data(X, shifts, max_shift):
    shifted = []
    for x, shift in zip(X, shifts):
        x = x[shift : max_shift - shift]
        shifted.append(x)
    return np.stack(shifted)


def infer(
    client: InferenceClient,
    callback: Callback,
    fname: Path,
    shifts: List[int],
    ifos: List[str],
    max_shift: int,
    sample_rate: float,
    inference_sampling_rate: float,
    batch_size: int,
    throughput: float,
    sequence_id: int,
):
    step_size = int(sample_rate / inference_sampling_rate)
    sleep = batch_size / inference_sampling_rate / throughput

    *x, t = read_timeseries(fname, *ifos)
    x = shift_data(x, shifts, max_shift)

    start, stop = t[0], t[-1] + t[1] - [0]
    num_steps = callback.start_new_sequence(start, stop)
    for i in range(num_steps):
        start = i * step_size * batch_size
        stop = (i + 1) * step_size * batch_size
        update = x[:, start:stop]
        client.infer(
            update,
            request_id=i,
            sequence_id=sequence_id,
            sequence_start=i == 0,
            sequence_end=i == (num_steps - 1),
        )

        while (i == 0) and not callback.initialized:
            time.sleep(1e-3)
        time.sleep(sleep)


@scriptify
def main(
    ip: str,
    model_name: str,
    data_dir: Path,
    write_dir: Path,
    sample_rate: float,
    batch_size: int,
    inference_sampling_rate: float,
    window_length: float,
    max_shift: float,
    throughput: float,
    sequence_id: int,
    model_version: int = -1,
    log_file: Optional[Path] = None,
    verbose: bool = False,
) -> None:
    configure_logging(log_file, verbose)

    matches = re.findall(r"([HLVK])([0-9]+\.[0-9])", data_dir.name)
    ifos, shifts = zip(**matches)
    ifos = [i + "1" for i in ifos]
    shifts = [int(float(i) * sample_rate) for i in shifts]
    max_shift = int(max_shift * sample_rate)

    callback = Callback(
        write_dir, inference_sampling_rate, batch_size, window_length
    )
    client = InferenceClient(
        f"{ip}:8001", model_name, model_version, callback=callback
    )
    with client:
        seqs_submitted, seqs_completed = 0, 0
        for fname in data_dir.iterdir():
            infer(
                client,
                callback,
                fname,
                shifts,
                ifos,
                max_shift,
                sample_rate,
                inference_sampling_rate,
                batch_size,
                throughput,
                sequence_id,
            )

            seqs_submitted += 1
            while client.get() is not None:
                seqs_completed += 1

        while seqs_completed < seqs_submitted:
            if client.get() is not None:
                seqs_completed += 1


if __name__ == "__main__":
    main()
