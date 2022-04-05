import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from bbhnet import io
from bbhnet.analysis import matched_filter


def analyze_segment(
    segment: io.timeslides.Segment,
    shift_dir: Path,
    write_dir: Path,
    window_length: float = 1.0,
    norm_seconds: Optional[float] = None,
):
    parents = set([Path(i).parents[2] for i in segment.fnames])
    if len(parents) > 1:
        raise ValueError("Too many parents!")
    parent = parents[0]

    fnames = [i.replace(parent, shift_dir) for i in segment.fnames]
    t, y, mf = matched_filter.analyze_segment(fnames, norm_seconds)
    fname = io.write_data(write_dir, t, y, mf)
    return fname


@contextmanager
def impatient_pool(num_proc):
    try:
        ex = ProcessPoolExecutor(num_proc)
        yield ex
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def analyze_outputs_parallel(
    segment: io.timeslides.Segment,
    data_dir: Path,
    write_dir: Path,
    window_length: float = 1.0,
    num_proc: int = 1,
    shifts: Optional[List[float]] = None,
    fnames: Optional[List[Path]] = None,
    norm_seconds: Optional[float] = None,
):
    ex = ProcessPoolExecutor(num_proc)
    futures = []
    with impatient_pool(num_proc) as ex:
        shifts = shifts or os.listdir(data_dir)
        for shift in shifts:
            timeslide = io.TimeSlide(data_dir / shift)
            for run in timeslide.runs:
                future = ex.submit(
                    analyze_segment,
                    segment,
                    run.path,
                    os.path.join(write_dir, shift),
                    window_length,
                    norm_seconds,
                )
                futures.append(future)

        for future in as_completed(futures):
            exc = future.exception()
            if isinstance(exc, FileNotFoundError):
                continue
            elif exc is not None:
                raise exc

            yield future.result()


def build_background(
    segment: io.timeslides.Segment,
    data_dir: str,
    write_dir: str,
    num_bins: int,
    window_length: float = 1.0,
    num_proc: int = 1,
    norm_seconds: Optional[float] = None,
    max_tb: Optional[float] = None,
):
    min_mf = float("inf")
    max_mf = -min_mf
    fnames, length = [], 0
    shifts = [i for i in os.listdir(data_dir) if i != "dt-0.0"]
    analyzer = analyze_outputs_parallel(
        segment,
        data_dir,
        write_dir,
        window_length=window_length,
        num_proc=num_proc,
        shifts=shifts,
        norm_seconds=norm_seconds,
    )

    Tb = 0
    with tqdm(total=max_tb) as pbar:
        for fname in analyzer:
            fnames.append(fname)
            minmax = io.minmax_re.search(fname)
            min_mf = min(min_mf, float(minmax.group("min")))
            max_mf = max(max_mf, float(minmax.group("max")))

            length = io.timeslides.fname_re.search(fname).group("length")
            pbar.update(length)
            Tb += length
            if Tb > max_tb:
                break
    return fnames, Tb, min_mf, max_mf
