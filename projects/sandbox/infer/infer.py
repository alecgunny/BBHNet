import logging
import re
import time
from dataclasses import dataclass
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


class ExistingSequence(Exception):
    pass


@dataclass
class Sequence:
    start: float
    stop: float
    sample_rate: float
    batch_size: int

    def __post_init__(self):
        num_predictions = (self.stop - self.start) * self.sample_rate
        num_steps = int(num_predictions // self.batch_size)
        num_predictions = int(num_steps * self.batch_size)

        self.predictions = np.zeros((num_predictions,))
        self.num_steps = num_steps
        self.initialized = False

    def update(self, y: np.ndarray, request_id: int) -> bool:
        self.initialized = True

        y = y[:, 0]
        start = request_id * self.batch_size
        stop = request_id * self.batch_size
        self.predictions[start:stop] = y
        return (request_id + 1) == self.num_steps

    @property
    def duration(self) -> float:
        return self.sample_rate * len(self.predictions)

    @property
    def fname(self) -> str:
        return "out_{self.start}-{self.duration}.hdf5"

    def __str__(self) -> str:
        return "{self.start}-{self.stop}"


class Callback:
    """
    Callable class for handling asynchronous server
    responses for streaming inference across sequences
    of timeseries data. New sequences should be
    initialized by calling the `start_new_sequence`
    method before starting to submit requests. Only
    one sequence can be inferred upon at once.

    Once inference has completed for each sequence,
    it will be asynchronously convolved with a
    boxcar filter to perform local integration, then
    both the raw and integrated timeseries will be
    written to a file in `write_dir` with the filename
    `out_{start}-{stop}.hdf5`, where `start` and `stop`
    indicate the initial and final GPS timestamps of
    the sequence.

    Args:
        write_dir:
            Directory to which to save network outputs
            in HDF5 format.
        inference_sampling_rate:
            Rate at which to sample inference windows
            from the input timeseries. Represents the
            sample rate of the output timeseries.
        batch_size:
            The number of subsequent windows to
            include in a single batch of inference.
        window_length:
            Length of the window over which network
            outputs should be locally integrated,
            specified in seconds.
    """

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
        self._sequence = None

    def start_new_sequence(self, start: float, stop: float) -> int:
        if self._sequence is not None:
            raise ExistingSequence(
                "Can't start inference on sequence of "
                "data from {}-{}, already managing "
                "sequence {}".format(start, stop, self._sequence)
            )

        self._sequence = Sequence(
            start, stop, self.inference_sampling_rate, self.batch_size
        )
        return self._sequence.num_steps

    def _make_timeseries(self, x: np.ndarray) -> TimeSeries:
        return TimeSeries(
            x,
            sample_rate=self.inference_sampling_rate,
            t0=self._sequence.start,
        )

    def integrate(self, y: np.ndarray) -> TimeSeries:
        """
        Convolve predictions with boxcar filter
        to get local integration, slicing off of
        the last values so that timeseries represents
        integration of _past_ data only.
        "Full" convolution means first few samples are
        integrated with 0s, so will have a lower magnitude
        than they technically should.
        """
        integrated = convolve(y, self.window, mode="full")
        integrated = integrated[: -self.window_size + 1]
        return self._make_timeseries(integrated)

    def write(self, tsdict: TimeSeriesDict):
        fname = self.write_dir / self._sequence.fname
        tsdict.write(fname, format="hdf5")
        logging.info(f"Wrote data to file {fname}")
        return fname

    def flush(self):
        """
        Integrate and write out current timeseries
        of predictions then clear internal state.
        """
        predictions = self._sequence.predictions
        integrated = self.integrate(predictions)
        predictions = self._make_timeseries(predictions)

        tsdict = TimeSeriesDict({"raw": predictions, "integrated": integrated})
        fname = self.write(tsdict)

        self._sequence = None
        return fname

    def __call__(self, y, request_id, sequence_id):
        # check to see if we've initialized a new
        # blank output array
        if self._sequence is None:
            raise SequenceNotStarted(
                "Must initialize sequence {} by calling "
                "`Callback.start_new_sequence` before "
                "submitting inference requests.".format(sequence_id)
            )
        if self._sequence.update(y, request_id):
            return self.flush()


def shift_data(
    X: List[np.ndarray], shifts: List[int], max_shift: int
) -> np.ndarray:
    """Time shift interferometer channels"""

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
) -> None:
    """Submit inference requests for a particular sequence

    Use a client which is currently connected to
    a Triton server instance to make inference
    requests for the data from a given filename
    with its channels time-shifted relative to
    one another.

    Args:
        client: The client to use for making requests
        callback:
            The callback object that manages how
            server responses are aggregated and written.
        fname:
            The HDF5 file to load timeseries data from.
        shifts:
            Relative shifts of each interferometer channel
            in units of samples.
        ifos:
            The names of the interferometer channels
            to load from the indicated `fname`.
        max_shift:
            The maximum shift value across all runs
            in the timeslide analysis that this run
            is a part of. This helps keep all output
            timeseries the same length.
        sample_rate:
            The rate at which the input timeseries
            data has been sampled.
        inference_sampling_rate:
            The rate at which to sample windows for
            inference from the input timeseries.
            Corresponds to the sample rate of the
            output timeseries.
        batch_size:
            The number of subsequent windows to
            include in a single batch of inference.
        throughput:
            Rate at which to make requests, in units
            of seconds of data per second `[s' / s]`.
        sequence_id:
            Identifier to assign to this sequence of
            inference requests to match up with a
            corresponding snapshot state on the
            inference server.
    """

    step_size = int(sample_rate / inference_sampling_rate)
    sleep = batch_size / inference_sampling_rate / throughput

    *x, t = read_timeseries(fname, *ifos)
    x = shift_data(x, shifts, max_shift)
    t = t[:-max_shift]

    start, stop = t[0], t[-1] + t[1] - t[0]
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
    inference_sampling_rate: float,
    batch_size: int,
    window_length: float,
    max_shift: float,
    throughput: float,
    sequence_id: int,
    model_version: int = -1,
    log_file: Optional[Path] = None,
    verbose: bool = False,
) -> None:
    """
    Perform inference using Triton on a directory
    of timeseries files, using a particular set of
    interferometer time shifts. Network outputs will
    be saved both as-is and using local integration.

    Args:
        ip:
            The IP address at which a Triton server
            hosting the indicated model is running
        model_name:
            The name of the model to which to make
            inference requests
        data_dir:
            Directory containing input files representing
            timeseries on which to perform inference.
            Each HDF5 file in this directory will be used
            for inference.
        write_dir:
            Directory to which to save raw and locally
            integrated network outputs.
        sample_rate:
            Rate at which input timeseries data has
            been sampled.
        inference_sampling_rate:
            The rate at which to sample windows for
            inference from the input timeseries.
            Corresponds to the sample rate of the
            output timeseries.
        batch_size:
            The number of subsequent windows to
            include in a single batch of inference.
        window_length:
            Length of the window over which network
            outputs should be locally integrated,
            specified in seconds.
        max_shift:
            The maximum shift value across all runs
            in the timeslide analysis that this run
            is a part of. This helps keep all output
            timeseries the same length.
        throughput:
            Rate at which to make requests, in units
            of seconds of data per second `[s' / s]`.
        sequence_id:
            Identifier to assign to all the sequences
            of inference requests in this run to match up
            with a corresponding snapshot state on the
            inference server.
        model_version:
            Version of the model from which to request
            inference. Default value of `-1` indicates
            the latest available version of the model.
        log_file:
            File to which to write inference logs.
        verbose:
            Flag controlling whether logging verbosity
            is `DEBUG` (`True`) or `INFO` (`False`)
    """
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
