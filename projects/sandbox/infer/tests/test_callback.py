from math import isclose
from unittest.mock import Mock

import numpy as np
import pytest
from infer.callback import Callback, ExistingSequence


@pytest.mark.parametrize(
    "integration_window_length,cluster_window_length", [(1, 8), (2, 8), (1, 4)]
)
class TestCallback:
    @pytest.fixture
    def callback(self, integration_window_length, cluster_window_length):
        return Callback(integration_window_length, cluster_window_length)

    def test_register(self, callback):
        assert callback.sequence is None
        sequence = Mock()
        callback.register(sequence)
        assert callback.sequence is sequence

        with pytest.raises(ExistingSequence):
            callback.register(sequence)

    def test_integrate(self, callback, integration_window_length):
        sample_rate = 256
        y = np.arange(sample_rate * 10)
        integrated = callback.integrate(y, sample_rate)
        assert len(integrated) == len(y)

        window_size = int(integration_window_length * sample_rate)
        for i, value in enumerate(integrated):
            if i < window_size:
                expected = sum(range(i + 2)) / window_size
            else:
                expected = (i + 3 + i - window_size) / 2
            assert isclose(value, expected, rel_tol=1e-9)
