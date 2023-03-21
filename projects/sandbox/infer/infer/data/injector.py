from dataclasses import dataclass

from bbhnet.analysis.ledger.injections import LigoResponseSet


@dataclass
class Injector:
    injection_set: LigoResponseSet
    start: float
    sample_rate: float

    def __call__(self, x):
        x_inj = self.injection_set.inject(x, self.start)
        self.start += x.shape[-1]
        return x, x_inj
