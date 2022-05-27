#!/usr/bin/env python3
# coding: utf-8
import os
import shutil
from pathlib import Path

from typing import TYPE_CHECKING, Optional, Tuple
import h5py
import pytest
import numpy as np


from bbhnet.io import h5

#from h5 import write_timeseries
#from h5 import read_timeseries

@pytest.fixture(scope="session")
def write_dir():
    write_dir = "tmp"
    os.makedirs(write_dir, exist_ok=True)
    yield Path(write_dir)
    shutil.rmtree(write_dir)

t = [1238163456,12345677890, 12345677890, 12345677890, 12345677890, 1238167552]
y = [23, 53.5, 234, 0, 5]
datasets = { 'dataset1': {'field_1': 'test1', 'field_2': 'test2', 'field_3': 'test3'},
             'dataset2': {'field_4': 'test4', 'field_5': 'test5', 'field_6': 'test6'}}
prefix = "out"

def test_write_timeseries(write_dir:"Path"):
    
    tmpdir = write_dir 
    fname = h5.write_timeseries(tmpdir, prefix, t, y, **datasets)

    with h5py.File(fname, "r") as f:

        #check the file name format
        length = t[-1] - t[0]
        assert fname == tmpdir / f"{prefix}_{t[0]}-{length}.hdf5"

        #check the file contents
        assert int(f["GPSstart"][0]) == t[0]

        for element in f["GPSstart"]:
            assert element in t

        for element in f["out"]:
            assert element in y, f"{element} not found"
        
        for dataset in datasets:
                assert  dataset in str(f.keys())

    #test the function without the optional parameter y, while reading check that its not written to the file

    fname = h5.write_timeseries(tmpdir, prefix, t, **datasets)
    with h5py.File(fname, "r") as f:
        assert f.get("out") == None

def test_read_timeseries(write_dir:"Path"):
    tmpdir = write_dir 

    length = t[-1] - t[0]
    fname = tmpdir / f"{prefix}_{t[0]}-{length}.hdf5"
    
    values = h5.read_timeseries(fname, *datasets)
    with h5py.File(fname, "r") as f:
        for key, value in zip(datasets, values[:-1]):
                assert np.asarray(f[key]) == value
        for element in f["GPSstart"]:
            assert element in np.asarray(values[-1])
        
