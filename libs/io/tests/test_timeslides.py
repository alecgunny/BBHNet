from unittest.mock import patch

import numpy as np
import pytest

from bbhnet.io.h5 import write_timeseries
from bbhnet.io.timeslides import Segment


@pytest.fixture
def t0():
    return 1234567890


@pytest.fixture
def timeslide_dir(tmpdir):
    timeslide_dir = tmpdir / "dt-0.0" / "0" / "out"
    timeslide_dir.mkdir(parents=True, exist_ok=False)
    return timeslide_dir


@pytest.fixture(params=[1024, 4096])
def file_length(request):
    return request.param


@pytest.fixture
def sample_rate():
    return 4


@pytest.fixture
def segment_fnames(timeslide_dir, t0, file_length, sample_rate):
    fnames = []
    num_samples = sample_rate * file_length
    for i in range(3):
        start = t0 + i * file_length
        t = np.arange(start, start + file_length, 1 / sample_rate)
        y = np.arange(i * num_samples, (i + 1) * num_samples)
        other = -y
        fname = write_timeseries(timeslide_dir, t=t, y=y, other=other)
        fnames.append(fname)
    return fnames


def test_segment(segment_fnames, t0, file_length, sample_rate):
    segment = Segment(segment_fnames)

    # test that all the properties got initialized correctly
    assert segment.t0 == t0
    assert segment.length == (file_length * 3)
    assert segment.shift == "dt-0.0"

    # make sure the __contains__ method works right
    for t in range(t0, t0 + segment.length):
        assert t in segment

    # test segment loading
    y, t = segment.load("out")
    expected_length = sample_rate * file_length * 3
    assert len(y) == len(t) == expected_length
    assert (y == np.arange(expected_length)).all()
    assert (t == np.arange(t0, t0 + file_length * 3, 1 / sample_rate)).all()

    # make sure cache elements are set correctly
    assert "out" in segment._cache
    assert "t" in segment._cache

    with patch("bbhnet.io.timeslides.Segment.read") as mock:
        file_size = sample_rate * file_length
        mock.return_value = y[:file_size], t[:file_size]
        y, t = segment.load("out")
        mock.assert_not_called()

        assert len(y) == len(t) == expected_length
        assert (y == np.arange(expected_length)).all()
        assert (
            t == np.arange(t0, t0 + file_length * 3, 1 / sample_rate)
        ).all()

        # loading another field though should need
        # to make a call to read_timeseries
        other, t = segment.load("other")
        mock.assert_called_with(str(segment_fnames[-1]), "other")

    # make sure that this other field also gets loaded right
    segment._cache.pop("other")
    other, t = segment.load("other")
    assert (other == -y).all()

    # TODOs:
    #   test make_shift
    #   test iteration
    #   test appending, including errors
