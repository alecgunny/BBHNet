import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Union

fname_re = re.compile(r"(?P<t0>\d{10}\.*\d*)-(?P<length>\d+\.*\d*).hdf5$")


def filter_and_sort_files(
    fnames: Union[str, Iterable[str]], return_matches: bool = False
):
    """Find all timestamped data files and sort them by their timestamps"""
    if isinstance(fnames, (Path, str)):
        fnames = os.listdir(fnames)

    # use the timestamps from all valid timestamped
    # filenames to sort the files as the first index
    # in a tuple
    matches = zip(map(fname_re.search, map(str, fnames)), fnames)
    tups = [(m.group("t0"), f, m) for m, f in matches if m is not None]

    # if return_matches is True, return the match object,
    # otherwise just return the raw filename
    return_idx = 2 if return_matches else 1
    return [t[return_idx] for t in sorted(tups)]


class Segment:
    fnames: Union[str, Iterable[str]]

    def __post_init__(self):
        if isinstance(self.fnames, (str, Path)):
            self.fnames = [self.fnames]

        matches = filter_and_sort_files(self.fnames, return_matches=True)
        self.t0 = float(matches[0].group("t0"))
        self.length = sum([float(i.group("length")) for i in matches])
        self.fnames = [i.string for i in matches]

        self._i = None

    @property
    def tf(self):
        return self.t0 + self.length

    def __contains__(self, timestamp):
        return self.t0 <= timestamp < self.tf

    def add(self, match: Union[str, re.Match]):
        if isinstance(match, str):
            match = fname_re.search(match)
            if match is None:
                raise ValueError(
                    f"Filename '{match}' not properly formatted "
                    "for addition to timeslide segment."
                )

        if float(match.group("t0")) != self.tf:
            raise ValueError(
                "Can't add file '{}' to run with files {}".format(
                    match.string, self.fnames
                )
            )

        self.fnames.append(match.string)
        self.length += float(match.group("length"))

    def __len__(self):
        return len(self.fnames)

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        if self._i is None:
            self.__iter__()
        elif self._i == len(self):
            self._i = None
            raise StopIteration

        fname = self.fnames[self._i]
        self._i += 1
        return fname


@dataclass
class Run:
    timeslide: "TimeSlide"
    index: int

    @property
    def path(self):
        return self.timeslide.path / str(self.index) / "out"

    def __lt__(self, other):
        if not isinstance(other, Run):
            raise TypeError(
                "Can't compare types '{}' and 'Run'".format(type(other))
            )
        return other.index < self.index


@dataclass
class TimeSlide:
    path: str

    def __post_init__(self):
        self.path = Path(self.path)
        self.runs = sorted([Run(self, int(i)) for i in os.listdir(self.path)])

        self.segments = []
        segment = None
        for run in self.runs:
            fnames = [run.path / i for i in os.listdir(run.path)]
            for match in filter_and_sort_files(fnames, return_matches=True):
                if segment is None:
                    segment = Segment(match.string)
                    continue

                try:
                    segment.add(match)
                except ValueError:
                    self.segments.append(segment)
                    segment = Segment(match.string)
        self.segments.append(segment)
