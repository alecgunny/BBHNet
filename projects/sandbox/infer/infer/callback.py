from dataclasses import dataclass

import numpy as np
from infer.sequence import Sequence

from bbhnet.analysis.events import TimeSlideEventSet


class SequenceNotStarted(Exception):
    pass


class ExistingSequence(Exception):
    pass


@dataclass
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

    integration_window_length: float
    cluster_window_length: float

    def integrate(self, y: np.ndarray, sample_rate: float) -> np.ndarray:
        """
        Convolve predictions with boxcar filter
        to get local integration, slicing off of
        the last values so that timeseries represents
        integration of _past_ data only.
        "Full" convolution means first few samples are
        integrated with 0s, so will have a lower magnitude
        than they technically should.
        """
        window_size = self.integration_window_length * sample_rate
        window = np.ones(window_size) / window_size
        integrated = np.convolve(y, window, mode="full")
        return integrated[: -window_size + 1]

    def cluster(
        self, y: np.ndarray, t0: float, sample_rate: float
    ) -> TimeSlideEventSet:
        window_size = int(self.cluster_window_length * sample_rate / 2)
        i = np.argmax(y[:window_size])
        events, times = [], []
        while i < len(y):
            val = y[i]
            window = y[i + 1 : i + 1 + window_size]
            if any(val <= window):
                i += np.argmax(window) + 1
            else:
                events.append(val)
                t = t0 + i * self.sample_rate
                times.append(t)
                i += self.window_size + 1
        return TimeSlideEventSet(events, times)

    def register(self, sequence: Sequence) -> int:
        if self.sequence is not None:
            raise ExistingSequence(
                "Can't start inference on sequence {} "
                "already managing sequence {}".format(sequence, self._sequence)
            )
        self.sequence = sequence

    def __call__(self, y, request_id, sequence_id):
        # check to see if we've initialized a new
        # blank output array
        if self.sequence is None:
            raise SequenceNotStarted(
                "Must initialize sequence {} by calling "
                "`Callback.start_new_sequence` before "
                "submitting inference requests.".format(sequence_id)
            )
        if self.sequence.update(y, request_id, sequence_id):
            y = self.sequence.predictions[sequence_id]
            integrated = self.integrate(y, self.sequence.sample_rate)
            return sequence_id, self.cluster(
                integrated, self.sequence.start, self.sequence.sample_rate
            )
