from unittest.mock import patch

import numpy as np
import pytest

from bbhnet.analysis.ledger import injections


class TestLigoResponseSet:
    @pytest.fixture
    def duration(self):
        return 4

    @pytest.fixture
    def sample_rate(self):
        return 128

    @pytest.fixture
    def N(self):
        return 10

    @pytest.fixture
    def ligo_response_set(self, duration, sample_rate, N):
        size = int(duration * sample_rate)
        params = {}
        bad_waveforms = {}
        waveforms = {}

        fields = injections.LigoResponseSet.__dataclass_fields__
        for name, attr in fields.items():
            if attr.metadata["kind"] == "parameter":
                params[name] = np.zeros((N,))
            elif attr.metadata["kind"] == "waveform":
                waveforms[name] = np.ones((N, size))
                bad_waveforms[name] = np.ones((N, size // 2))

        with pytest.raises(ValueError) as exc:
            kwargs = {}
            kwargs.update(params)
            kwargs.update(bad_waveforms)
            injections.LigoResponseSet(
                sample_rate=sample_rate,
                duration=duration,
                num_injections=N,
                **kwargs
            )
        assert str(exc.value).startswith("Specified waveform duration")

        with pytest.raises(ValueError) as exc:
            kwargs = {}
            kwargs.update(params)
            kwargs.update(waveforms)
            injections.LigoResponseSet(
                sample_rate=sample_rate,
                duration=duration,
                num_injections=N - 1,
                **kwargs
            )
        assert str(exc.value).startswith("LigoResponseSet")

        return injections.LigoResponseSet(
            sample_rate=sample_rate,
            duration=duration,
            num_injections=N,
            **kwargs
        )

    def test_waveforms(self, ligo_response_set, sample_rate, duration, N):
        size = int(sample_rate * duration)
        expected_shape = (N, 2, size)

        assert ligo_response_set._waveforms is None
        assert ligo_response_set.waveforms.shape == expected_shape
        assert ligo_response_set._waveforms.shape == expected_shape
        with patch("numpy.stack") as mock:
            ligo_response_set.waveforms
        mock.assert_not_called()

    def test_get_times(self, ligo_response_set, N):
        ligo_response_set.gps_time = np.arange(N)
        with pytest.raises(ValueError):
            ligo_response_set.get_times()

        obj = ligo_response_set.get_times(start=2)
        assert len(obj) == 8
        assert obj.num_injections == N

        obj = ligo_response_set.get_times(end=6)
        assert len(obj) == 6
        assert obj.num_injections == N

        obj = ligo_response_set.get_times(2, 6.5)
        assert len(obj) == 5
        assert obj.num_injections == N

    def test_read(self, ligo_response_set, tmp_path, N):
        tmp_path.mkdir(exist_ok=True)
        fname = tmp_path / "obj.h5"

        ligo_response_set.gps_time = np.arange(N)
        ligo_response_set.write(fname)

        # TODO: generalize logic here to duration
        new = injections.LigoResponseSet.read(fname, start=2.5)
        assert len(new) == N - 1
        assert (new.gps_time == np.arange(1, N)).all()

        new = injections.LigoResponseSet.read(fname, end=6)
        assert len(new) == N - 1
        assert (new.gps_time == np.arange(N - 1)).all()

        new = injections.LigoResponseSet.read(fname, start=3.5, end=5)
        assert len(new) == 6
        assert (new.gps_time == np.arange(2, 8)).all()

    def test_append(self, ligo_response_set, N):
        ligo_response_set.gps_time = np.arange(N)
        ligo_response_set.waveforms

        new = ligo_response_set[:6]
        new.num_injections = 13
        new.gps_time = np.arange(N, N + 6)
        new.h1 *= 2
        new.l1 *= 2

        ligo_response_set.append(new)
        assert ligo_response_set.num_injections == N + 13
        assert (ligo_response_set.gps_time == np.arange(N + 6)).all()
        assert (ligo_response_set.h1[N:] == 2).all()
        assert (ligo_response_set.l1[N:] == 2).all()
        assert ligo_response_set._waveforms is None

    def test_inject(self, ligo_response_set, sample_rate, duration, N):
        ligo_response_set.gps_time = (duration + 1) * np.arange(10)
        ligo_response_set.h1 += np.arange(N)[:, None]
        ligo_response_set.l1 += np.arange(N)[:, None]
        ligo_response_set.l1 *= -1

        start = -1
        length = 27
        x = np.zeros((2, length * sample_rate))
        y = ligo_response_set.inject(x, start)
        assert (y[0, : (duration - 1) * sample_rate] == 1).all()
        assert (y[1, : (duration - 1) * sample_rate] == -1).all()

        offset = (duration - 1) * sample_rate
        for i in range(5):
            zero_start = offset + i * (duration + 1) * sample_rate
            zero_end = zero_start + sample_rate
            assert not y[:, zero_start:zero_end].any()

            wave_start = zero_end
            wave_end = wave_start + duration * sample_rate
            assert (y[0, wave_start:wave_end] == i + 2).all()
            assert (-y[1, wave_start:wave_end] == i + 2).all()

            if i == 4:
                assert wave_end == (x.shape[-1] + sample_rate)
