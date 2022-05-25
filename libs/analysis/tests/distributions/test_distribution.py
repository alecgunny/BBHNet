from math import exp
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from bbhnet.analysis.distributions.distribution import (
    SECONDS_IN_YEAR,
    Distribution,
)


def nb(x):
    try:
        return np.array([10 for _ in x])
    except TypeError:
        return 10


@pytest.fixture
def sample_rate():
    return 4


@pytest.fixture(params=[0, 1, -1])
def offset(request, sample_rate):
    return request.param / (2 * sample_rate)


@pytest.fixture
def event_time(offset):
    return SECONDS_IN_YEAR / 100 - offset


@pytest.fixture(params=[float, list])
def event_times(request, event_time, sample_rate):
    if request.param == float:
        return event_time
    else:
        return np.array([event_time + i * 10 for i in range(3)])


def test_distribution(event_time, event_times, offset, sample_rate):
    distribution = Distribution("test")
    distribution.nb = nb
    distribution.Tb = SECONDS_IN_YEAR / 2

    far = distribution.far(0)
    assert far == 20

    tf = (event_time + offset) * 2
    significance = distribution.significance(0, tf)
    assert significance == 1 - exp(-tf * 22 / SECONDS_IN_YEAR)

    t = np.arange(0, tf, 1 / sample_rate)
    y = np.ones_like(t)
    segment = Mock()
    segment.load = MagicMock(return_value=(y, t))

    for metric, expected in zip(["far", "significance"], [far, significance]):
        characterization, times = distribution.characterize_events(
            segment, event_times=event_times, window_length=1, metric=metric
        )
        segment.load.assert_called_with("test")

        if isinstance(event_times, float):
            assert characterization.ndim == 1
            assert len(characterization) == sample_rate

            assert times.ndim == 1
            start = int(len(t) // 2) + 1
            if offset > 0:
                start -= 1

            t_expect = t[start : start + sample_rate] - event_time
            assert np.isclose(times, t_expect, rtol=1e-9).all()
        else:
            assert characterization.shape == (3, sample_rate)
            assert times.shape == (3, sample_rate)

            assert times.ndim == 2
            assert len(times) == 3
            assert times.shape[-1] == sample_rate

            for i, tc in enumerate(event_times):
                start = int(len(t) // 2) + 1
                if offset > 0:
                    start -= 1
                start += i * 10 * sample_rate

                t_expect = t[start : start + sample_rate] - tc
                assert np.isclose(times[i], t_expect, rtol=1e-9).all()

        assert np.isclose(characterization, expected, rtol=1e-7).all()
