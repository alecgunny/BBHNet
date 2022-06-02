#!/usr/bin/env python3
# coding: utf-8
import os
import shutil
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pytest

from bbhnet.io import h5


@pytest.fixture(scope="function")
def write_dir():
    write_dir = "tmp"
    os.makedirs(write_dir, exist_ok=True)
    yield Path(write_dir)
    shutil.rmtree(write_dir)


@pytest.fixture(scope="function")
def t():
    t = [
        1238163456,
        12345677890,
        12345677890,
        12345677890,
        12345677890,
        1238167552,
    ]
    return t


@pytest.fixture(scope="function")
def y():
    y = [23, 53.5, 234, 0, 5, 456.345]
    return y


@pytest.fixture(scope="function")
def datasets():
    datasets = {
        "dataset1": np.arange(6),
        "dataset2": np.arange(10, 16),
        "dataset3": np.arange(20, 26),
        "dataset4": np.arange(30, 36),
    }
    return datasets


@pytest.fixture(scope="function")
def prefix():
    prefix = "out"
    return prefix


def check_file_contents(
    fname, t, prefix, y: Optional["np.ndarray"] = None, **datasets
):
    with h5py.File(fname, "r") as f:
        # check the timestamps array
        assert (t == f["GPSstart"][:]).all()

        # if there is "out" array, it should be checked
        if y:
            assert (y == f[prefix][:]).all()
            assert len(y) == len(f[prefix])

        for key, value in datasets.items():
            assert key in str(f.keys())
            assert len(f[key]) == len(t)
            assert (value == f[key][:]).all()


def test_write_timeseries(write_dir: "Path", t, y, prefix, datasets):

    fname = h5.write_timeseries(write_dir, prefix, t, y, **datasets)

    # check the file name format
    length = t[-1] - t[0]
    assert fname == write_dir / f"{prefix}_{t[0]}-{length}.hdf5"

    # check if file contents were written properly
    check_file_contents(fname, t, prefix, y, **datasets)

    # test the function without the optional parameter y
    # while reading check that its not written to the file
    fname = h5.write_timeseries(write_dir, prefix, t, **datasets)
    check_file_contents(fname, t, prefix, **datasets)


def test_read_timeseries(write_dir: "Path", t, y, prefix, datasets):

    length = t[-1] - t[0]
    fname = write_dir / f"{prefix}_{t[0]}-{length}.hdf5"

    # first manually write the timeseries in the required format
    with h5py.File(fname, "w") as f:
        f["GPSstart"] = t
        if y is not None:
            f["out"] = y

        for key, value in datasets.items():
            f[key] = value

    # test the read_timeseries function
    values = h5.read_timeseries(fname, *datasets)

    with h5py.File(fname, "r") as f:
        for key, value in zip(datasets, values[:-1]):
            assert (value == f[key][:]).all()

        # last array in "values" should be the timeseries
        assert (values[-1] == f["GPSstart"][:]).all()
