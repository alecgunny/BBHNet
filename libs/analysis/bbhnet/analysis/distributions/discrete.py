from dataclasses import dataclass

import numpy as np

from bbhnet.analysis.distributions.distribution import Distribution


@dataclass
class DiscreteDistribution(Distribution):
    mininum: float
    maximum: float
    num_bins: float
    clip: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        self.bins = np.linspace(self.mininum, self.maximum, self.num_bins + 1)
        self.histogram = np.zeros((self.num_bins,))

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
            nb = self.histogram[bins[:-1] >= threshold].sum(axis=0)
        print(threshold, nb)
        return nb

    def update(self, x: np.ndarray, t: np.ndarray):
        counts, _ = np.histogram(x, self.bins)
        if counts.sum() < len(x) and not self.clip:
            counts[0] += (x < self.minimum).sum()
            counts[-1] += (x >= self.maximum).sum()

        self.histogram += counts
        self.Tb += t[-1] - t[0]
