from dataclasses import dataclass
from typing import Iterable, Union

import numpy as np

from bbhnet.io.timeslides import Segment


@dataclass
class Distribution:
    dataset: str

    def __post_init__(self):
        self.Tb = 0

    def update(self, x: np.ndarray, t: np.ndarray):
        """Update this distribution to reflect new data"""

        raise NotImplementedError

    def nb(self, threshold: float):
        """Number of events in this distribution above a threshold"""

        raise NotImplementedError

    def far(self, threshold: float, analysis_time: float) -> float:
        """see https://arxiv.org/pdf/1508.02357.pdf, eq. 17"""

        nb = self.nb(threshold)
        # time_ratio = analysis_time / self.Tb
        # return 1 - np.exp(-time_ratio * (1 + nb))
        return 365 * 24 * 3600 * nb / self.Tb

    def characterize_events(
        self,
        segment: "Segment",
        event_times: Union[float, Iterable[float]],
        window_length: float = 1,
        kernel_length: float = 1,
    ):
        # duck-typing check on whether there are
        # multiple events in the segment or just the one.
        # Even if there's just one but it's passed as an
        # iterable, we'll record return a 2D array, otherwise
        # just return 1D
        try:
            event_iter = iter(event_times)
            single_event = False
        except TypeError:
            event_iter = iter([event_times])
            single_event = True

        y, t = segment.load(self.dataset)
        sample_rate = 1 / (t[1] - t[0])
        window_size = int(window_length * sample_rate)

        fars, times = [], []
        for event_time in event_iter:
            # start with the first timestep that could
            # have contained the event in the NN input
            idx = ((t - event_time) > 0).argmax()
            event_far = self.far(y[idx : idx + window_size], segment.length)

            fars.append(event_far)
            times.append(t[idx : idx + window_size] - event_time)

        fars, times = np.stack(fars), np.stack(times)
        if single_event:
            return fars[0], times[0]
        return fars, times

    def fit(
        self,
        segments: Union[Segment, Iterable[Segment]],
        warm_start: bool = True,
    ) -> None:
        if not warm_start:
            self.__post_init__()

        # TODO: accept pathlike and initialize a timeslide?
        if isinstance(segments, Segment):
            segments = [segments]

        for segment in segments:
            y, t = segment.load(self.dataset)
            self.update(y, t)

    def __str__(self):
        return f"{self.__class__.__name__}(Tb={self.Tb})"
