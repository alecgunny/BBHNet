from collections import defaultdict
from typing import Callable, Iterable, Optional, Tuple

import numpy as np

from bbhnet.analysis.analysis import integrate
from bbhnet.analysis.distributions import ClusterDistribution


def distribution_dict(ifos: Iterable[str], t_clust: float) -> defaultdict:
    def factory():
        bckgrd = ClusterDistribution("integrated", ifos, t_clust)
        frgrd = ClusterDistribution("integrated", ifos, t_clust)

        return {"background": bckgrd, "foreground": frgrd}

    return defaultdict(factory)


def integrate_and_fit(
    y: np.ndarray,
    t: np.ndarray,
    shifts: Iterable[str],
    distribution: dict,
    window_length: float,
    normalizer: Optional[Callable] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t, y, integrated = integrate(
        y, t, window_length=window_length, normalizer=normalizer
    )
    distribution.fit((integrated, t), shifts, warm_start=True)
    return t, y, integrated
