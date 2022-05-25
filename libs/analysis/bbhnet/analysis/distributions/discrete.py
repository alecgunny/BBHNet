import logging
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from bbhnet.analysis.distributions.distribution import Distribution


@dataclass
class DiscreteDistribution(Distribution):
    minimum: float
    maximum: float
    num_bins: float
    clip: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        self.bins = np.linspace(self.minimum, self.maximum, self.num_bins + 1)
        self.histogram = np.zeros((self.num_bins,))

    def write(self, path: Path):
        with h5py.File(path, "w") as f:
            f["bins"] = self.bins
            f["histogram"] = self.histogram
            f["fnames"] = list(map(str, self.fnames))
            f["Tb"] = self.Tb

    @property
    def bin_centers(self):
        return (self.bins[:-1] + self.bins[1:]) / 2

    def nb(self, threshold: float):
        # TODO: is this the best way to do this check?
        if isinstance(threshold, np.ndarray):
            bins = np.repeat(self.bins[:-1, None], len(threshold), axis=1)
            hist = np.repeat(self.histogram[:, None], len(threshold), axis=1)
            mask = bins >= threshold
            nb = (hist * mask).sum(axis=0)
        else:
            nb = self.histogram[self.bins[:-1] >= threshold].sum(axis=0)

        logging.debug(
            "Threshold {} has {} events greater than it "
            "in distribution {}".format(threshold, nb, self)
        )
        return nb

    def update(self, x: np.ndarray, t: np.ndarray):
        counts, _ = np.histogram(x, self.bins)
        if counts.sum() < len(x) and not self.clip:
            counts[0] += (x < self.bins[0]).sum()
            counts[-1] += (x >= self.bins[-1]).sum()

        self.histogram += counts
        self.Tb += t[-1] - t[0] + t[1] - t[0]
