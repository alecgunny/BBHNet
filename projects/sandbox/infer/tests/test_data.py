from unittest.mock import Mock, patch

import numpy as np
import pytest
from infer import data


def test_shift_chunk():
    x = np.arange(10)
    x = np.stack([x, x - 1], axis=0)
    y, rems = data._shift_chunk(x, [0, 1], remainders=[None, None])
    assert y.shape == (2, 9)
    assert (y[0] == y[1]).all()

    assert len(rems) == 2
    assert len(rems[0]) == 1
    assert len(rems[1]) == 0

    x = x + 10
    y, rems = data._shift_chunk(x, [0, 1], rems)
    assert y.shape == (2, 10)
    assert (y[0] == np.arange(9, 19)).all()
    assert (y[0] == y[1]).all()

    assert len(rems) == 2
    assert len(rems[0]) == 1
    assert len(rems[1]) == 0


class TestSegmentIterator:
    @pytest.fixture(params=[128])
    def sample_rate(self, request):
        return request.param

    @pytest.fixture
    def repeater(self, sample_rate):
        def it(N, step, shift):
            shift = int(sample_rate * shift)
            step = int(sample_rate * step)
            for i in range(N):
                x = step * i + np.arange(step)
                yield np.stack([x, x - shift])

        return it

    @pytest.fixture
    def validate_iteration(self, sample_rate, repeater):
        def f(shift, method):
            it = repeater(10, 10, shift)
            it = data.SegmentIterator(it, 0, 10, sample_rate, [0, shift])
            if method is not None:
                it = getattr(it, method)()

            length = 0
            for i, y in enumerate(it):
                expected_length = int(10 * sample_rate)
                if not i:
                    expected_length -= int(shift * sample_rate)

                expected = length + np.arange(expected_length)
                length += len(expected)
                assert (y == expected).all()
            assert i == 9

        return f

    def test_init(self, sample_rate):
        obj = data.SegmentIterator(
            Mock(), 1234567890, 1234567900, sample_rate, [0, 5]
        )
        # TODO: what else
        assert obj.duration == 10

    def test_shift_it(self, validate_iteration):
        validate_iteration(4, "_shift_it")

    def test_iter(self, validate_iteration):
        validate_iteration(5, None)


class TestSequence:
    @pytest.fixture(params=[8])
    def sample_rate(self, request):
        return request.param

    @pytest.fixture(params=[4, 8, 32])
    def batch_size(self, request):
        return request.param

    @patch("bbhnet.analysis.ledger.injections.LigoResponseSet.read")
    def test_init(self, mock, sample_rate, batch_size):
        segment = Mock()
        segment.start = 0
        segment.end = 129
        segment.duration = 129

        injection_set = Mock()
        obj = data.Sequence(
            segment, injection_set, sample_rate, batch_size, 10
        )
        mock.assert_called_with(injection_set, start=0, end=129)

        if batch_size > sample_rate:
            assert obj.background.size == (128 * sample_rate)
            assert obj.num_steps == 128 / 4
        else:
            assert obj.background.size == (129 * sample_rate)
            if batch_size == sample_rate:
                assert obj.num_steps == 129
            else:
                assert obj.num_steps == 129 * 2
