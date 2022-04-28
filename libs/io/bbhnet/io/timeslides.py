import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple, Union

import numpy as np

from bbhnet.io.h5 import read_timeseries

fname_re = re.compile(r"(?P<t0>\d{10}\.*\d*)-(?P<length>\d+\.*\d*).hdf5$")


def filter_and_sort_files(
    fnames: Union[str, Iterable[str]], return_matches: bool = False
):
    """Find all timestamped data files and sort them by their timestamps"""
    if isinstance(fnames, (Path, str)):
        fnames = Path(fnames)
        if not fnames.is_dir():
            raise ValueError(f"'{fnames}' is not a directory")

        # this is a generator of paths but it's fine
        # because the map(str, fnames) below makes
        # the iterator agnostic to this
        fnames = fnames.iterdir()

    # use the timestamps from all valid timestamped
    # filenames to sort the files as the first index
    # in a tuple
    matches = zip(map(fname_re.search, map(str, fnames)), fnames)
    tups = [(m.group("t0"), f, m) for m, f in matches if m is not None]

    # if return_matches is True, return the match object,
    # otherwise just return the raw filename
    return_idx = 2 if return_matches else 1
    return [t[return_idx] for t in sorted(tups)]


@dataclass
class Segment:
    fnames: Union[str, Iterable[str]]

    def __post_init__(self):
        if isinstance(self.fnames, (str, Path)):
            self.fnames = [self.fnames]

        for f in self.fnames:
            if not Path(f).exists():
                raise ValueError(f"Segment file {f} does not exist")

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

    def append(self, match: Union[str, re.Match]):
        """Add a new file to the end of this segment"""

        # make sure the filename is appropriate formatted
        if isinstance(match, str):
            match = fname_re.search(match)
            if match is None:
                raise ValueError(
                    f"Filename '{match}' not properly formatted "
                    "for addition to timeslide segment."
                )

        # make sure this filename starts off
        # where thesegment currently ends
        if float(match.group("t0")) != self.tf:
            raise ValueError(
                "Can't add file '{}' to run with files {}".format(
                    match.string, self.fnames
                )
            )

        # append the filename and increase the length accordingly
        self.fnames.append(match.string)
        self.length += float(match.group("length"))

    def shift(self, dirname: str) -> "Segment":
        """
        Create a new segment with the same filenames
        from a different timeslide.

        Args:
            dirname:
                The root directory of the new timeslide
                to map this Segment's filenames to
        """

        fnames = [Path(f) for f in self.fnames]
        new_fnames = []
        for fname in fnames:
            parts = list(fname.parts)
            parts[-4] = dirname
            new_fnames.append(Path("/").joinpath(*parts))
        return Segment(new_fnames)

    def load(self, *datasets) -> Tuple[np.ndarray, ...]:
        """Load the specified fields from this Segment's HDF5 files"""

        outputs = defaultdict(list)
        t = []
        for fname in self.fnames:
            values = read_timeseries(fname, *datasets)
            for key, value in zip(datasets, values[:-1]):
                outputs[key].append(value)
            t.append(values[-1])

        if len(self.fnames) > 1:
            outputs = {k: np.concatenate(v) for k, v in outputs.items()}
            t = np.concatenate(t)
        else:
            outputs = {k: v[0] for k, v in outputs.items()}
            t = t[0]

        return tuple(outputs[key] for key in datasets) + (t,)

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

    def __str__(self):
        root = Path(self.fnames[0]).parents[2]
        return f"Segment(root='{root}', t0={self.t0}, length={self.length})"


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
        self.runs = sorted(
            [Run(self, int(i.name)) for i in self.path.iterdir()]
        )

        self.segments = []
        segment = None
        for run in self.runs:
            fnames = [run.path / i for i in run.path.iterdir()]
            for match in filter_and_sort_files(fnames, return_matches=True):
                if segment is None:
                    segment = Segment(match.string)
                    continue

                try:
                    segment.append(match)
                except ValueError:
                    self.segments.append(segment)
                    segment = Segment(match.string)
        self.segments.append(segment)
