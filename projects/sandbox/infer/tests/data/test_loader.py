import h5py
import numpy as np
import pytest
from infer.data import loader


@pytest.fixture(params=[0, 1, 2])
def shift(request):
    return request.param


@pytest.fixture
def sample_rate():
    return 128


@pytest.fixture
def data_dir(shift, sample_rate, tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    x = np.arange(1024)

    for i in range(4):
        start = 1234567890 + i * 8
        duration = 4
        fname = tmp_path / f"tmp-{start}-{duration}.h5"
        with h5py.File(fname, "w") as f:
            f["H1"] = x
            f["L1"] = x - int(shift * sample_rate)
    return tmp_path


@pytest.fixture(params=[1, 2.5, 4])
def chunk_length(request):
    return request.param


def test_load_fname(data_dir, shift, chunk_length, sample_rate):
    fname = next(data_dir.iterdir())
    shifts = [0, int(shift * sample_rate)]
    chunk_size = int(chunk_length * sample_rate)

    num_chunks, rem = divmod(8 - shift, chunk_length)
    if rem:
        num_chunks += 1

    it = loader.load_fname(fname, ["H1", "L1"], shifts, chunk_size)
    outputs = []
    for i, x in enumerate(it):
        assert x.shape[0] == 2

        if (i + 1) < num_chunks:
            assert x.shape[-1] == chunk_size
        elif (i + 1) == num_chunks and not rem:
            assert x.shape[-1] == chunk_size
        elif (i + 1) == num_chunks:
            assert x.shape[-1] == int(rem * sample_rate)

        outputs.append(x)
    assert (i + 1) == num_chunks

    output = np.concatenate(outputs, axis=-1)
    expected_length = int(sample_rate * (8 - shift))
    assert output.shape == (2, expected_length)
    assert (output == np.arange(expected_length)).all()
