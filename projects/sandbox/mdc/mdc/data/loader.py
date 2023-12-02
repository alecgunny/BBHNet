from pathlib import Path
from typing import List

import h5py
import numpy as np


def aggregate(data):
    return np.stack(data).astype("float32")


def load_dataset(
    fs, dataset, channels: List[str], chunk_size: int
) -> np.ndarray:
    size = len(fs[0][channels[0]][dataset])
    num_steps = int(size // chunk_size)
    for i in range(num_steps):
        slc = slice(i * chunk_size, (i + 1) * chunk_size)
        xs = [[i[j][dataset][slc] for j in channels] for i in fs]
        xs = [aggregate(x) for x in xs]
        yield np.stack(xs)


def load_datasets(
    data_dir: Path, datasets: str, channels: List[str], chunk_size: int
) -> np.ndarray:
    bg_fname = data_dir / "background.hdf"
    fg_fname = data_dir / "foreground.hdf"

    fs = [h5py.File(f, "r") for f in [bg_fname, fg_fname]]
    try:
        for dataset in datasets:
            size = len(fs[0][channels[0]][dataset])
            it = load_dataset(fs, dataset, channels, chunk_size)
            yield (int(dataset), size), it
    finally:
        [f.close() for f in fs]
