import logging
import re
from concurrent.futures import FIRST_EXCEPTION, wait
from pathlib import Path
from typing import Iterable, List, Optional

from analyze import segment as segment_utils
from analyze.distribution import distribution_dict, integrate_and_fit
from rich.progress import Progress

from bbhnet.analysis.normalizers import GaussianNormalizer
from bbhnet.io.timeslides import Segment, TimeSlide
from bbhnet.logging import configure_logging
from bbhnet.parallelize import AsyncExecutor, as_completed
from hermes.typeo import typeo


def fit_distributions(
    read_pool: AsyncExecutor,
    write_pool: AsyncExecutor,
    pbar: Progress,
    background_segments: Iterable[Segment],
    foreground_field: str,
    data_dir: Path,
    write_dir: Path,
    max_tb: float,
    t_clust: float,
    window_length: float,
    norm_seconds: Optional[Iterable[float]] = None,
):
    norm_seconds = norm_seconds or [norm_seconds]
    ifos = ["H1", "L1"]
    initials = [i[0] for i in ifos]
    shift_pattern = re.compile(rf"(?<=[{initials}])[0-9\.]+")

    distributions = distribution_dict(ifos, t_clust)
    write_futures = []

    background_segments = iter(background_segments)
    main_task_id = pbar.add_task("[red]Building background", total=max_tb)
    while not pbar.tasks[main_task_id].finished:
        try:
            segment = next(background_segments)
        except StopIteration:
            break

        # iterate through all possible timeshifts of this zero-shifted
        # segment and load in both the background data as well as
        # any foreground injection data if it exists
        analysis_jobs, load_futures = 0, []
        for shift in segment.shift_dir.iterdir():
            background, foreground = segment_utils.find_shift_and_foreground(
                segment, shift.name, foreground_field
            )
            if background is None:
                continue

            future = read_pool.submit(
                segment_utils.load_segments, (background, foreground)
            )
            load_futures[shift.name] = [future]
            analysis_jobs += 1 if foreground is None else 2

        # create progress bar tasks for each one
        # of the subprocesses involved for analyzing
        # this set of timeslide ClusterDistributions
        load_task_id = pbar.add_task(
            f"[cyan]Loading {len(load_futures)} {segment.length}s timeslides",
            total=len(load_futures),
        )
        analyze_task_id = pbar.add_task(
            "[yelllow]Integrating timeslides",
            total=analysis_jobs * len(norm_seconds),
        )
        write_task_id = pbar.add_task(
            "[green]Writing integrated timeslides",
            total=analysis_jobs * len(norm_seconds),
        )

        def write_cb(f):
            pbar.update(write_task_id, advance=1)

        # as these loads complete, integrate and normalize the
        # background and foreground timeseries using each of
        # the specified normalization values and fit distributions
        # to the integrated values
        for shift, (yf, yb, t) in as_completed(load_futures):
            pbar.update(load_task_id, advance=1)
            sample_rate = 1 / (t[1] - t[0])

            shifts = shift_pattern.findall(shift)
            shifts = list(map(float, shifts))

            for norm in norm_seconds:
                # treat 0 the same as no normalization
                norm = norm or None
                dists = distributions[norm]

                # build a normalizer for the given normalization window length
                if norm is not None:
                    normalizer = GaussianNormalizer(norm * sample_rate)
                    normalizer.fit(yb)
                else:
                    normalizer = None

                # integrate the nn outputs and use them to fit
                # a background distribution. The `distributions`
                # dict will create a new distribution if one doesn't
                # already exist for this normalization value
                t_int, yb, int_b = integrate_and_fit(
                    yb,
                    t,
                    shifts,
                    dists["background"],
                    window_length,
                    normalizer,
                )
                pbar.update(analyze_task_id, advance=1)

                future = write_pool.submit(
                    segment_utils.write_segment,
                    write_dir,
                    shift,
                    field="background-integrated",
                    norm=norm,
                    t=t_int,
                    y=yb,
                    integrated=int_b,
                )
                future.add_done_callback(write_cb)
                write_futures.append(future)

                if yf is not None:
                    # do the same with the foreground data
                    # for this segment if it exists
                    _, yf, int_f = integrate_and_fit(
                        yf,
                        t,
                        shifts,
                        dists["foreground"],
                        window_length,
                        normalizer,
                    )
                    pbar.update(analyze_task_id, advance=1)

                    future = write_pool.submit(
                        segment_utils.write_segment,
                        write_dir,
                        shift,
                        field="foreground-integrated",
                        norm=norm,
                        t=t_int,
                        y=yf,
                        integrated=int_f,
                    )
                    future.add_done_callback(write_cb)
                    write_futures.append(future)

            tb = t_int[-1] - t_int[0] + 1 / sample_rate
            pbar.update(main_task_id, advance=tb)

    Tb = pbar.tasks[main_task_id].completed
    logging.info(f"Accumulated {Tb}s of background matched filter outputs.")

    # write all of the background distributions
    for norm, dists in distributions.items():
        for dist_type, dist in dists.items():
            dist.write(write_dir / f"{dist_type}_{norm}.h5")
    return distributions, write_futures


@typeo
def main(
    data_dir: Path,
    write_dir: Path,
    results_dir: Path,
    t_clust: float,
    window_length: Optional[float] = None,
    norm_seconds: Optional[List[float]] = None,
    max_tb: Optional[float] = None,
    force: bool = False,
    log_file: Optional[str] = None,
    verbose: bool = False,
):
    """Analyze injections in a directory of timeslides

    Iterate through a directory of timeslides analyzing known
    injections for false alarm rates in units of yrs$^{-1}$ as
    a function of the time after the event trigger times enters
    the neural network's input kernel. For each event and normalization
    period specified by `norm_seconds`, use time- shifted data from
    segments _before_ the event's segment tobuild up a background
    distribution of the output of matched filters of length `window_length`,
    normalized by the mean and standard deviation of the previous
    `norm_seconds` worth of data, until the effective time analyzed
    is equal to `max_tb`.
    The results of this analysis will be written to two csv files,
    one of which will contain the latency and false alaram rates
    for each of the events and normalization windows, and the other
    of which will contain the bins and counts for the background
    distributions used to calculate each of these false alarm rates.
    Args:
        data_dir: Path to directory containing timeslides and injections
        write_dir: Path to directory to which to write matched filter outputs
        results_dir:
            Path to directory to which to write analysis logs and
            summary csvs for analyzed events and their corresponding
            background distributions.
        window_length:
            Length of time, in seconds, over which to average
            neural network outputs for matched filter analysis
        t_clust: Clustering timescale for background distributions
        norm_seconds:
            Length of time, in seconds, over which to compute a moving
            "background" used to normalize the averaged neural network
            outputs. More specifically, the matched filter output at each
            point in time will be the average over the last `window_length`
            seconds, normalized by the mean and standard deviation of the
            previous `norm_seconds` seconds. If left as `None`, no
            normalization will be performed. Otherwise, should be specified
            as an iterable to compute multiple different normalization values
            for each event.
        max_tb:
            The maximum number of time-shifted background data to analyze
            per event, in seconds
        force:
            Flag indicating whether to force an event analysis to re-run
            if its data already exists in the summary files written to
            `results_dir`.
        log_file:
            A filename to write logs to. If left as `None`, logs will only
            be printed to stdout
        verbose:
            Flag indicating whether to log at level `INFO` (if set)
            or `DEBUG` (if not set)
    """

    results_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(log_file, verbose)

    # initiate process and thread pools
    read_pool = AsyncExecutor(4, thread=False)
    write_pool = AsyncExecutor(4, thread=True)

    # organize background and injection timeslides into segments
    zero_shift = data_dir / "dt-H0.0-L0.0"
    background_ts = TimeSlide(zero_shift, field="background-out")
    background_segments = background_ts.segments

    with read_pool, write_pool, Progress() as pbar:
        backgrounds, write_futures = fit_distributions(
            read_pool,
            write_pool,
            pbar,
            background_segments,
            foreground_field="injection-out",
            data_dir=data_dir,
            write_dir=write_dir,
            max_tb=max_tb,
            t_clust=t_clust,
            window_length=window_length,
            norm_seconds=norm_seconds,
        )
        wait(write_futures, return_when=FIRST_EXCEPTION)


if __name__ == "__main__":
    main()
