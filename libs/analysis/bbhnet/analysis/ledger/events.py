from dataclasses import dataclass
from typing import List, TypeVar

import numpy as np

from bbhnet.analysis.ledger.injections import InterferometerResponseSet
from bbhnet.analysis.ledger.ledger import Ledger, metadata, parameter

SECONDS_IN_YEAR = 31556952
F = TypeVar("F", np.ndarray, float)


@dataclass
class TimeSlideEventSet(Ledger):
    detection_statistic: np.ndarray = parameter()
    time: np.ndarray = parameter()
    Tb: float = metadata(default=0)

    def compare_metadata(self, key, ours, theirs):
        if key == "Tb":
            return ours + theirs
        return super().compare_metadata(key, ours, theirs)

    def nb(self, threshold: F) -> F:
        try:
            len(threshold)
        except TypeError:
            return (self.detection_statistic >= threshold).sum()
        else:
            stats = self.detection_statistic[:, None]
            return (stats >= threshold).sum(0)

    def far(self, threshold: F) -> F:
        nb = self.nb(threshold)
        return SECONDS_IN_YEAR * nb / self.Tb

    def significance(self, threshold: F, T: float) -> F:
        """see https://arxiv.org/pdf/1508.02357.pdf, eq. 17
        Represents the likelihood that at least one event
        with detection statistic value `threshold` will occur
        after observing this distribution for a period `T`.
        Args:
            threshold: The detection statistic to compare against
            T:
                The length of the analysis period in which the
                detection statistic was measured, in seconds
        """

        nb = self.nb(threshold)
        return 1 - np.exp(-T * (1 + nb) / self.Tb)


@dataclass
class EventSet(TimeSlideEventSet):
    shift: np.ndarray = parameter()

    def get_shift(self, shift):
        mask = self.shift == shift
        if self.shift.ndim == 2:
            mask = mask.all(axis=-1)

        # TODO: should we return TimeSlideEventSet?
        return self[mask]

    @classmethod
    def from_timeslide(cls, event_set: TimeSlideEventSet, shift: List[float]):
        shifts = np.array([shift] * len(event_set))
        d = {k: getattr(event_set, k) for k in event_set.__dataclass_fields__}
        return cls(shift=shifts, **d)


# inherit from TimeSlideEventSet since injection
# will already have shift information recorded
@dataclass
class RecoveredInjectionSet(TimeSlideEventSet, InterferometerResponseSet):
    @classmethod
    def recover(
        cls,
        events: TimeSlideEventSet,
        responses: InterferometerResponseSet,
        offset: float,
    ):
        # TODO: need an implementation that will
        # also do masking on shifts
        diffs = np.abs(events.time - responses.gps_time[:, None] - offset)
        idx = diffs.argmin(axis=-1)
        events = events[idx]
        kwargs = events.__dict__ | responses.__dict__
        for attr in responses.waveform_fields:
            kwargs.pop(attr)
        return cls(**kwargs)
