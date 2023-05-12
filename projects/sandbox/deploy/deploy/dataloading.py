import logging
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from gwpy.timeseries import TimeSeries

PATH_LIKE = Union[str, Path]

patterns = {
    "prefix": "[a-zA-Z0-9_:-]+",
    "start": "[0-9]{10}",
    "duration": "[1-9][0-9]*",
    "suffix": "(gwf)|(hdf5)|(h5)",
}
groups = {k: f"(?P<{k}>{v})" for k, v in patterns.items()}
pattern = "{prefix}-{start}-{duration}.{suffix}".format(**groups)
fname_re = re.compile(pattern)


def parse_frame_name(fname: PATH_LIKE) -> Tuple[str, int, int]:
    """Use the name of a frame file to infer its initial timestamp and length

    Expects frame names to follow a standard nomenclature
    where the name of the frame file ends {prefix}_{timestamp}-{length}.gwf

    Args:
        fname: The name of the frame file
    Returns:
        The prefix of the frame file name
        The initial GPS timestamp of the frame file
        The length of the frame file in seconds
    """

    if isinstance(fname, Path):
        fname = fname.name

    match = fname_re.search(fname)
    if match is None:
        raise ValueError(f"Could not parse frame filename {fname}")

    prefix, start, duration, *_ = match.groups()
    return prefix, int(start), int(duration)


def _is_gwf(match):
    return match is not None and match.group("suffix") == "gwf"


def get_prefix(data_dir: Path):
    if not data_dir.exists():
        raise FileNotFoundError(f"No data directory '{data_dir}'")

    fnames = map(str, data_dir.iterdir())
    matches = map(fname_re.search, fnames)
    matches = list(filter(_is_gwf, matches))

    if len(matches) == 0:
        raise ValueError(f"No valid .gwf files in data directory '{data_dir}'")

    t0 = min([int(i.group("start")) for i in matches])
    prefixes = set([i.group("prefix") for i in matches])
    if len(prefixes) > 1:
        raise ValueError(
            "Too many prefixes {} in data directory '{}'".format(
                list(prefixes), data_dir
            )
        )

    durations = set([i.group("duration") for i in matches])
    if len(durations) > 1:
        raise ValueError(
            "Too many lengths {} in data directory '{}'".format(
                list(durations), data_dir
            )
        )
    return list(prefixes)[0], int(list(durations)[0]), t0


class MissingFrame(Exception):
    pass


def data_iterator(
    data_dir: Path,
    channel: str,
    ifos: List[str],
    sample_rate: float,
    timeout: Optional[float] = None,
) -> torch.Tensor:
    prefix, length, t0 = get_prefix(data_dir / ifos[0])
    middle = prefix.split("_")[1]

    # give ourselves a little buffer so we don't
    # try to grab a frame that's been filtered out
    t0 += length * 2
    while True:
        frames = []
        logging.debug(f"Reading frames from timestamp {t0}")

        for ifo in ifos:
            prefix = f"{ifo[0]}-{ifo}_{middle}"
            fname = data_dir / ifo / f"{prefix}-{t0}-{length}.gwf"
            tick = time.time()
            while not fname.exists():
                tock = time.time()
                if timeout is not None and (tock - tick > timeout):
                    raise MissingFrame(
                        "Couldn't find frame file {} after {}s".format(
                            fname, timeout
                        )
                    )
            x = read_channel(fname, f"{ifo}:{channel}", sample_rate)
            frames.append(x)

        logging.debug("Read successful")
        yield torch.Tensor(np.stack(frames)), t0
        t0 += length


def resample(x: TimeSeries, sample_rate: float):
    if x.sample_rate.value != sample_rate:
        return x.resample(sample_rate)
    return x


def read_channel(fname, channel, sample_rate):
    for i in range(3):
        try:
            x = TimeSeries.read(fname, channel=channel)
        except ValueError as e:
            if str(e) == (
                "Cannot generate TimeSeries with 2-dimensional data"
            ):
                logging.warning(
                    "Channel {} from file {} got corrupted and was "
                    "read as 2D, attempting reread {}".format(
                        channel, fname, i + 1
                    )
                )
                time.sleep(1e-1)
                continue
            else:
                raise
        except RuntimeError as e:
            if str(e).startswith("Failed to read the core"):
                logging.warning(
                    "Channel {} from file {} had corrupted header, "
                    "attempting reread {}".format(channel, fname, i + 1)
                )
                time.sleep(2e-1)
                continue
            else:
                raise

        x = resample(x, sample_rate)
        if len(x) != sample_rate:
            logging.warning(
                "Channel {} in file {} got corrupted with "
                "length {}, attempting reread {}".format(
                    channel, fname, len(x), i + 1
                )
            )
            del x
            time.sleep(1e-1)
            continue

        return x
    else:
        raise ValueError(
            "Failed to read channel {} in file {}".format(channel, fname)
        )
