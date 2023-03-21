import logging
import re
import sys
import time
import traceback
from multiprocessing import Event, Process, Queue
from pathlib import Path
from queue import Empty, Full
from typing import List, Optional

import h5py
import numpy as np


def load_fname(
    fname: Path, channels: List[str], shifts: List[int], chunk_size: int
) -> np.ndarray:
    max_shift = max(shifts)
    with h5py.File(fname, "r") as f:
        size = len(f[channels[0]])
        idx = 0
        while idx < size:
            data = []
            for channel, shift in zip(channels, shifts):
                start = idx + shift
                stop = start + chunk_size

                # make sure that segments with shifts shorter
                # than the max shift get their ends cut off
                stop = min(size - (max_shift - shift), stop)

                x = f[channel][start:stop]
                data.append(x)

            yield np.stack(data).astype("float32")
            idx += chunk_size


def _loader(
    data_dir: Path,
    channels: List[str],
    chunk_length: float,
    sample_rate: float,
    shifts: Optional[List[float]],
    event: Event,
):
    fname_re = re.compile(r"(?P<t0>\d{10}\.*\d*)-(?P<length>\d+\.*\d*)")
    file_it = data_dir.iterdir()
    chunk_size = int(chunk_length * sample_rate)

    if shifts is not None:
        max_shift = max(shifts)
        shifts = [int(i * sample_rate) for i in shifts]
    else:
        max_shift = 0
        shifts = [0 for _ in channels]

    while not event.is_set():
        try:
            fname = next(file_it)
        except StopIteration:
            event.set()
            break

        match = fname_re.search(fname.name)
        if match is None:
            continue

        # first return some information about
        # the segment we're about to iterate through
        start = float(match.group("t0"))
        duration = float(match.group("length"))
        yield (start, start + duration - max_shift)

        # now iterate through the segment in chunks
        yield from load_fname(fname, channels, shifts, chunk_size)

        # now return None to indicate this segment is done
        yield None

    # finally a back-to-back None to indicate
    # that all segments are completed
    yield None


def target(
    data_dir: Path,
    channels: List[str],
    chunk_length: float,
    sample_rate: float,
    shifts: Optional[List[float]],
    event,
    q,
):
    def try_put(x):
        while not event.is_set():
            try:
                q.put_nowait(x)
            except Full:
                time.sleep(1e-3)
            else:
                break

    try:
        it = _loader(
            data_dir, channels, chunk_length, sample_rate, shifts, event
        )
        while not event.is_set():
            x = next(it)
            try_put(x)
    except Exception:
        exc_type, exc, tb = sys.exc_info()
        tb = traceback.format_exception(exc_type, exc, tb)
        tb = "".join(tb[:-1])
        try_put((exc_type, str(exc), tb))
    finally:
        # if we arrived here from an exception, i.e.
        # the event has not been set, then don't
        # close the queue until the parent process
        # has received the error message and set the
        # event itself, otherwise it will never be
        # able to receive the message from the queue
        try:
            q.get_nowait()
        except Empty:
            pass

        while not event.is_set():
            time.sleep(1e-3)
        q.close()


def load_data(
    data_dir: Path,
    channels: List[str],
    chunk_length: float,
    sample_rate: float,
    shifts: Optional[List[float]] = None,
) -> np.ndarray:
    if shifts is not None and len(shifts) != len(channels):
        raise ValueError(
            "Specified {} shifts but {} channels".format(
                len(shifts), len(channels)
            )
        )
    q = Queue(1)
    event = Event()
    args = (data_dir, channels, chunk_length, sample_rate, shifts, event, q)

    p = Process(target=target, args=args)
    p.start()

    def try_get():
        while not event.is_set():
            try:
                x = q.get_nowait()
            except Empty:
                time.sleep(1e-3)
                continue

            if isinstance(x, tuple) and len(x) == 3:
                exc_type, msg, tb = x
                logging.exception(
                    "Encountered exception in data collection process:\n" + tb
                )
                raise exc_type(msg)
            return x

    def gen():
        while True:
            x = try_get()
            if x is None:
                break
            yield x

    try:
        while True:
            x = try_get()
            if x is None:
                break
            start, stop = x
            yield (start, stop), gen()
    finally:
        event.set()
        q.close()
        p.join()
        p.close()
