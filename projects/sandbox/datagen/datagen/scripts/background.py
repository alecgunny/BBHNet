import logging
from pathlib import Path
from typing import List

import h5py
import mldatafind.io as io
from gwpy.timeseries import TimeSeriesDict
from mldatafind.find import find_data
from mldatafind.segments import query_segments
from typeo import scriptify

from bbhnet.logging import configure_logging


def _intify(x: float):
    return int(x) if int(x) == x else x


# TODO: add to mldatafind.io
def fname_from_ts_dict(ts: TimeSeriesDict, prefix: str):
    times = ts.times
    length = times[-1] - times[0] + times[1] - times[0]
    t0 = times[0]

    t0 = _intify(t0)
    length = _intify(length)
    fname = f"{prefix}-{t0}-{length}.hdf5"
    return fname


def check_cache(
    outdir: Path,
    train_start: float,
    train_stop: float,
    ifos: List[str],
    minimum_length: float,
    sample_rate: float,
) -> bool:
    """
    Returns True if data is cached as expected
    """
    files = io.filter_and_sort_files(outdir, train_start, train_stop)

    # if no files are found we need to generate data
    try:
        training_file = files[0]
    except IndexError:
        return False

    # if there exists files in training range,
    # check the timestamp and verify that it
    # meets the requested conditions
    with h5py.File(training_file, "r") as f:
        missing_keys = [i for i in ifos if i not in f]
        if missing_keys:
            raise ValueError(
                "Background file {} missing data from {}".format(
                    training_file, ", ".join(missing_keys)
                )
            )

        t0 = f.attrs["t0"][()]
        length = len(f[ifos[0]]) / sample_rate

    in_range = train_start <= t0 <= (train_stop - minimum_length)
    long_enough = length >= minimum_length
    if in_range and long_enough:
        return True
    else:
        raise ValueError(
            "Background file {} has t0 {} and length {}s, "
            "which isn't compatible with request of {}s "
            "segment between {} and {}".format(
                training_file,
                t0,
                length,
                minimum_length,
                train_start,
                train_stop,
            )
        )


@scriptify
def main(
    train_start: float,
    train_stop: float,
    test_stop: float,
    minimum_train_length: float,
    minimum_test_length: float,
    ifos: List[str],
    sample_rate: float,
    channel: str,
    frame_type: str,
    state_flag: str,
    datadir: Path,
    logdir: Path,
    force_generation: bool = False,
    verbose: bool = False,
):
    """Generates background data for training and testing BBHnet

    Args:
        start: start gpstime
        stop: stop gpstime
        ifos: which ifos to query data for
        outdir: where to store data
    """
    # make logdir dir
    logdir.mkdir(exist_ok=True, parents=True)
    datadir.mkdir(exist_ok=True, parents=True)
    configure_logging(logdir / "generate_background.log", verbose)

    # check cache for training segment
    # TODO: how should we check cache for testing segments?
    # should we wait until we've queried segments and then
    # check that those segments exist?
    generate_data = True
    if not force_generation:
        cached = check_cache(
            datadir,
            train_start,
            train_stop,
            ifos,
            minimum_train_length,
            sample_rate,
        )
        generate_data = not cached

    if not generate_data:
        logging.info(
            "Background data already exists and forced "
            "generation is off. Not generating background"
        )
        return

    # first query segments that meet minimum length
    # requirement during the requested training period
    train_segments = query_segments(
        [f"{ifo}:{state_flag}" for ifo in ifos],
        train_start,
        train_stop,
        minimum_train_length,
    )

    try:
        train_segment = train_segments[0]
    except IndexError:
        raise ValueError(
            "No segments of minimum length, not producing background"
        )

    # now query segments that meet testing requirements
    # and append training segment
    segments = query_segments(
        [f"{ifo}:{state_flag}" for ifo in ifos],
        train_stop,
        test_stop,
        minimum_test_length,
    )
    segments.append(train_segment)

    channels = [f"{ifo}:{channel}" for ifo in ifos]
    iterator = find_data(
        segments,
        channels,
    )

    for data in iterator:

        # resample and write
        data = data.resample(sample_rate)
        file_path = fname_from_ts_dict(data, prefix="background")
        data.write(file_path)

    return datadir
