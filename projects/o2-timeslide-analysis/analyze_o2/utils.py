from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Optional

from tqdm import tqdm

from bbhnet.analysis.distributions import DiscreteDistribution
from bbhnet.analysis.matched_filter import analyze_segment
from bbhnet.io.timeslides import Segment


@contextmanager
def impatient_pool(num_proc):
    try:
        ex = ProcessPoolExecutor(num_proc)
        yield ex
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def analyze_segments_parallel(
    segments: Iterable[Segment],
    write_dir: Path,
    window_length: float = 1.0,
    num_proc: int = 1,
    shifts: Optional[Iterable[float]] = None,
    norm_seconds: Optional[float] = None,
):
    ex = ProcessPoolExecutor(num_proc)
    futures = []
    with impatient_pool(num_proc) as ex:
        for segment in segments:
            # TODO: logic to extract shift from segment.path.parents
            shift = None
            future = ex.submit(
                analyze_segment,
                segment,
                window_length=window_length,
                kernel_length=1.0,
                norm_seconds=norm_seconds,
                write_dir=write_dir / shift,
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
    segment: Segment,
    write_dir: str,
    window_length: float = 1.0,
    num_proc: int = 1,
    norm_seconds: Optional[float] = None,
    max_tb: Optional[float] = None,
    num_bins: int = int(1e4),
):
    # TODO: do some logic to figure out how many
    # shifts we need to do on this segment to hit
    # max_tb, then create segments using the same
    # base filenames but with different shifts.
    # Possibly wrap this into a `.shift` method of
    # segments that maps up to the write parent level
    # and replaces with the corresponding dirname
    # shifts = [i for i in os.listdir(data_dir) if i != "dt-0.0"]
    segments = []
    analyzer = analyze_segments_parallel(
        segments,
        write_dir,
        window_length=window_length,
        num_proc=num_proc,
        norm_seconds=norm_seconds,
    )

    # keep track of the min and max values so that
    # we can initialize the bins of a discrete distribution
    # to get the maximum level of resolution possible
    min_mf, max_mf, fnames = float("inf"), -float("inf"), []
    for fname, minval, maxval in tqdm(analyzer, total=len(segments)):
        fnames.append(fname)
        min_mf = min(min_mf, minval)
        max_mf = max(max_mf, maxval)

    # TODO: best way to infer num bins?
    # TODO: should we parallelize the background fit?
    background = DiscreteDistribution(min_mf, max_mf, num_bins)
    background.fit(list(map(Segment, fnames)))
    return background
