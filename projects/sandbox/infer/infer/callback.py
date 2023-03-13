import logging
from dataclasses import dataclass

import numpy as np
from infer.data import Sequence

from bbhnet.analysis.ledger.events import (
    RecoveredInjectionSet,
    TimeSlideEventSet,
)


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

    id: int
    integration_window_length: float
    cluster_window_length: float
    fduration: float

    def __post_init__(self):
        self._sequence = None
        self.offset = self.integration_window_length - self.fduration / 2

    @property
    def sequence(self):
        if self._sequence is None:
            raise SequenceNotStarted(
                f"No sequence associated with sequence id '{self.id}'"
            )
        return self._sequence

    @sequence.setter
    def sequence(self, new):
        if self._sequence is not None and new is not None:
            raise ExistingSequence(
                "Can't start inference on sequence {} "
                "already managing sequence {}".format(new, self._sequence)
            )
        self._sequence = new

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
        window_size = int(self.integration_window_length * sample_rate)
        window = np.ones(window_size) / window_size
        integrated = np.convolve(y, window, mode="full")
        return integrated[: -window_size + 1]

    def cluster(self, y) -> TimeSlideEventSet:
        sample_rate = self.sequence.sample_rate
        t0 = self.sequence.segment.start

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
                t = t0 + i / sample_rate
                times.append(t)
                i += window_size + 1

        Tb = len(y) / sample_rate
        events = np.array(events)
        times = np.array(times)
        return TimeSlideEventSet(events, times, Tb)

    def register(self, sequence: Sequence) -> int:
        if self.sequence is not None:
            raise ExistingSequence(
                "Can't start inference on sequence {} "
                "already managing sequence {}".format(sequence, self._sequence)
            )
        self.sequence = sequence

    def postprocess(self, y):
        y = self.integrate(y, self.sequence.sample_rate)
        return self.cluster(y)

    def __call__(self, y, request_id, sequence_id):
        # check to see if we've initialized a new
        # blank output array
        if self.sequence is None:
            raise SequenceNotStarted(
                "Must initialize sequence {} by calling "
                "`Callback.start_new_sequence` before "
                "submitting inference requests.".format(sequence_id)
            )
        if sequence_id == self.id:
            self.sequence.background.update(y)
        else:
            self.sequence.foreground.update(y)

        if self.sequence.done:
            logging.debug(f"Finished inference on sequence {self.sequence}")

            background_events = self.postprocess(self.sequence.background.y)
            foreground_events = self.postprocess(self.sequence.foreground.y)
            foreground_events = RecoveredInjectionSet.recover(
                foreground_events,
                self.sequence.injection_set,
                int(self.offset * self.sequence.sample_rate),
            )

            logging.debug(f"Finished postprocessing sequence {self.sequence}")
            self.sequence = None
            return background_events, foreground_events
