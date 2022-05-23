import logging
from collections import defaultdict
from concurrent.futures import FIRST_EXCEPTION, wait
from itertools import product
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from hermes.typeo import typeo
from rich.progress import Progress

from bbhnet.analysis.analysis import integrate
from bbhnet.analysis.distributions import DiscreteDistribution
from bbhnet.analysis.normalizers import GaussianNormalizer
from bbhnet.io.h5 import write_timeseries
from bbhnet.io.timeslides import Segment, TimeSlide
from bbhnet.logging import configure_logging
from bbhnet.parallelize import AsyncExecutor, as_completed

event_times = [1186302519.8, 1186741861.5, 1187058327.1, 1187529256.5]
event_names = ["GW170809", "GW170814", "GW170818", "GW170823"]
events = {name: time for name, time in zip(event_names, event_times)}


# define some dummy functions that just execute some
# method of an object, since you can't submit an object's
# method directly to a process pool because it's not picklable
def load_segment(segment: Segment):
    segment.load("out")
    return segment


def fit(background: DiscreteDistribution, fnames: Iterable[str]):
    background.fit(list(map(Segment, fnames)))
    return background


# now define the meat of our functions
def get_write_dir(
    write_dir: Path, norm: Optional[float], shift: Union[str, Segment]
) -> Path:
    if isinstance(shift, Segment):
        shift = shift.shift

    write_dir = write_dir / f"norm-seconds.{norm}" / shift
    write_dir.mkdir(parents=True, exist_ok=True)
    return write_dir


def build_background(
    thread_ex: AsyncExecutor,
    process_ex: AsyncExecutor,
    pbar: Progress,
    background_segments: Iterable[Segment],
    data_dir: Path,
    write_dir: Path,
    max_tb: float,
    window_length: float = 1.0,
    norm_seconds: Optional[Iterable[float]] = None,
    num_bins: int = int(1e4),
):
    write_dir.mkdir(exist_ok=True)

    norm_seconds = norm_seconds or [norm_seconds]
    normalizers = {}
    for norm in norm_seconds:
        # TODO: infer this dynamically when we load the segments
        sample_rate = 16
        if norm is not None:
            normalizers[norm] = GaussianNormalizer(norm * sample_rate)
        else:
            normalizers[norm] = None

    # keep track of the min and max values of each normalization
    # window's background and the corresponding filenames so
    # that we can fit a discrete distribution to it after the fact
    mins = defaultdict(lambda: float("inf"))
    maxs = defaultdict(lambda: -float("inf"))
    fnames = defaultdict(list)

    # iterate through timeshifts of our background segments
    # until we've generated enough background data.
    background_segments = iter(background_segments)
    main_task_id = pbar.add_task("[red]Building background", total=max_tb)
    while not pbar.tasks[main_task_id].finished:
        segment = next(background_segments)

        # since we're assuming here that the background
        # segments are being provided in reverse chronological
        # order (with segments closest to the event segment first),
        # exhaust all the time shifts we can of each segment before
        # going to the previous one to keep data as fresh as possible
        load_futures = {}
        for shift in data_dir.iterdir():
            try:
                shifted = segment.make_shift(shift.name)
            except ValueError:
                # this segment doesn't have a shift
                # at this value, so just move on
                continue

            # load all the timeslides up front in a separate thread
            # TODO: O(1GB) memory means segment.length * N ~O(4M),
            # so for ~O(10k) long segments this means this should
            # be fine as long as N ~ O(100). Worth doing a check for?
            future = thread_ex.submit(load_segment, shifted)
            load_futures[shift.name] = [future]

        # create progress bar tasks for each one
        # of the subprocesses involved for analyzing
        # this set of timeslides
        load_task_id = pbar.add_task(
            f"[cyan]Loading {len(load_futures)} {segment.length}s timeslides",
            total=len(load_futures),
        )
        analyze_task_id = pbar.add_task(
            "[yelllow]Integrating timeslides",
            total=len(load_futures) * len(norm_seconds),
        )
        write_task_id = pbar.add_task(
            "[green]Writing integrated timeslides",
            total=len(load_futures) * len(norm_seconds),
        )

        # now once each segment is loaded, submit a job
        # to our process pool to integrate it using each
        # one of the specified normalization periods
        integration_futures = {}
        for shift, seg in as_completed(load_futures):
            pbar.update(load_task_id, advance=1)
            for norm in norm_seconds:
                future = process_ex.submit(
                    integrate,
                    seg,
                    kernel_length=1.0,
                    window_length=window_length,
                    normalizer=normalizers[norm],
                )
                future.add_done_callback(
                    lambda f: pbar.update(analyze_task_id, advance=1)
                )
                integration_futures[(norm, shift)] = [future]

        # as the integration jobs come back, write their
        # results using our thread pool and record the
        # min and max values for our discrete distribution
        fname_futures = []
        for (norm, shift), (t, y, integrated) in as_completed(
            integration_futures
        ):
            # submit the writing job to our thread pool and
            # use a callback to keep track of all the filenames
            # for a given normalization window
            shift_dir = get_write_dir(write_dir, norm, shift)
            future = thread_ex.submit(
                write_timeseries,
                shift_dir,
                t=t,
                y=y,
                integrated=integrated,
            )
            future.add_done_callback(lambda f: fnames[norm].append(f.result()))
            future.add_done_callback(
                lambda f: pbar.update(write_task_id, advance=1)
            )
            fname_futures.append(future)

            # keep track of the max and min values for each norm
            mins[norm] = min(mins[norm], integrated.min())
            maxs[norm] = max(maxs[norm], integrated.max())

        wait(fname_futures, return_when=FIRST_EXCEPTION)
        pbar.update(main_task_id, advance=len(load_futures) * segment.length)

    Tb = pbar.tasks[main_task_id].completed
    logging.info(f"Accumulated {Tb}s of background matched filter outputs.")

    # initialize a discrete background distribution with the
    # bounds we found during iteration, then fit it to all
    # the matched filter outputs we just analyzed in separate
    # processes for each background distribution.
    # TODO: best way to infer num bins?
    bkgrd_task_id = pbar.add_task(
        "[red]Fitting background distributions", total=len(norm_seconds)
    )

    fit_futures = {}
    for norm, fs in fnames.items():
        mn, mx = mins[norm], maxs[norm]
        background = DiscreteDistribution("integrated", mn, mx, num_bins)

        future = process_ex.submit(fit, background, fs)
        future.add_done_callback(
            lambda f: pbar.update(bkgrd_task_id, advance=1)
        )
        fit_futures[norm] = [future]

    # Wait for each of these distributions to fit before
    # returning the background distributions.
    # TODO: could potentially do this in a callback but
    # probably good to have some logging here.
    backgrounds = {}
    for norm, bacgkround in as_completed(fit_futures):
        backgrounds[norm] = background
        logging.info(
            "Background for norm_seconds={} fit "
            "with {}s worth of data".format(norm, background.Tb)
        )
    return backgrounds


def check_if_needs_analyzing(
    event_segment: Segment,
    norm_seconds: Iterable[Optional[float]],
    characterizations: pd.DataFrame,
) -> Iterable[Optional[float]]:
    times = [t for t in event_times if t in event_segment]
    names = [name for name in event_names if events[name] in times]

    combos = set(product(names, norm_seconds))
    remaining = combos - set(characterizations.index)

    # only do analysis on those normalization
    # values that we haven't already done
    # (sorry, you'll still have to do it for all events,
    # but those are miniscule by comparison)
    norm_seconds = list(set([j for i, j in remaining]))
    return norm_seconds, names, times


def analyze_event(
    thread_ex: AsyncExecutor,
    process_ex: AsyncExecutor,
    characterizations: pd.DataFrame,
    timeseries: pd.DataFrame,
    event_segment: Segment,
    background_segments: Iterable[Segment],
    data_dir: Path,
    write_dir: Path,
    results_dir: Path,
    max_tb: float,
    window_length: float = 1.0,
    norm_seconds: Optional[Iterable[float]] = None,
    num_bins: int = int(1e4),
    force: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Use timeshifts of a set of previous segments to build a
    background distribution with which to analyze a segment
    containing an event and characterizing the false alaram
    rate of that event as a function of time from the event
    trigger.
    """

    # first check if we can skip this analysis altogether
    # because we already have data on it and we're not
    # forcing ourselves to re-analyze
    norm_seconds = norm_seconds or [norm_seconds]
    if not force:
        norm_seconds, names, times = check_if_needs_analyzing(
            event_segment, norm_seconds, characterizations
        )
        if len(norm_seconds) == 0:
            logging.info(
                f"Already analyzed events in segment {event_segment}, skipping"
            )
            return

    # TODO: exclude segments with events?
    with Progress() as pbar:
        backgrounds = build_background(
            thread_ex,
            process_ex,
            pbar,
            background_segments=background_segments,
            data_dir=data_dir,
            write_dir=write_dir,
            window_length=window_length,
            norm_seconds=norm_seconds,
            max_tb=max_tb,
            num_bins=num_bins,
        )

    # now use the fit background to characterize the
    # significance of BBHNet's detection around the event
    for norm, background in backgrounds.items():
        if norm is not None:
            normalizer = GaussianNormalizer(norm)
        else:
            normalizer = None

        logging.info(
            "Characterizing events {} with normalization "
            "window length {}".format(", ".join(names), norm)
        )
        t, y, integrated = integrate(
            event_segment,
            kernel_length=1,
            window_length=window_length,
            normalizer=normalizer,
        )
        fname = write_timeseries(
            get_write_dir(write_dir, norm, event_segment),
            t=t,
            y=y,
            integrated=integrated,
        )

        # create a segment and add the existing data to
        # its cache so that we don't try to load it again
        segment = Segment(fname)
        segment._cache = {"t": t, "integrated": integrated}
        fars, latencies = background.characterize_events(
            segment, event_times, window_length=window_length, metric="far"
        )

        # for each one of the events in this segment,
        # record the false alarm rate as a function of
        # time and add it to our dataframe then checkpoint it.
        # Then isolate the timeseries of both the NN outputs and
        # the integrated values around the event and write those
        # to another dataframe and checkpoint that as well
        for far, latency, name, time in zip(fars, latencies, names, times):
            logging.info(f"\t{name}:")
            logging.info(f"\t\tFalse Alarm Rates: {list(far)}")
            logging.info(f"\t\tLatencies: {list(latency)}")

            df = pd.DataFrame(
                dict(
                    event_name=[name] * len(far),
                    norm_seconds=[norm] * len(far),
                    far=far,
                    latency=latency,
                )
            ).set_index(["event_name", "norm_seconds"])
            characterizations = pd.concat([characterizations, df])
            characterizations.to_csv(results_dir / "characterizations.csv")

            # keep the one second before the trigger,
            # during the event after the trigger, and
            # after the event trigger has left the kernel
            mask = (time - 1 < t) & (t < time + 2)
            df = pd.DataFrame(
                dict(
                    event_name=[name] * mask.sum(),
                    norm_seconds=[norm] * len(far),
                    t=t[mask] - time,
                    y=y[mask],
                    integrated=integrated[mask],
                )
            ).set_index(["event_name", "norm_seconds"])
            timeseries = pd.concat([timeseries, df])
            timeseries.to_csv(results_dir / "timeseries.csv")

        # write an h5 file describing the background distribution
        fname = "background_events.{}_norm.{}.hdf5".format(
            names.join(","), norm
        )
        background.write(results_dir / fname)

    return far, t, background


@typeo
def main(
    data_dir: Path,
    write_dir: Path,
    results_dir: Path,
    window_length: float = 1.0,
    norm_seconds: Optional[List[float]] = None,
    max_tb: Optional[float] = None,
    num_bins: int = 10000,
    force: bool = False,
    log_file: Optional[str] = None,
    verbose: bool = False,
):
    """Analyze known events in a directory of timeslides

    Iterate through a directory of timeslides analyzing known
    events for false alarm rates in units of yrs$^{-1}$ as
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
        data_dir: Path to directory contains timeslides
        write_dir: Path to directory to which to write matched filter outputs
        results_dir:
            Path to directory to which to write analysis logs and
            summary csvs for analyzed events and their corresponding
            background distributions.
        window_length:
            Length of time, in seconds, over which to average
            neural network outputs for matched filter analysis
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
        num_bins:
            The number of bins to use in building up the discrete background
            distribution
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
    configure_logging(results_dir / log_file, verbose)

    # organize timeslides into segments
    timeslide = TimeSlide(data_dir / "dt-0.0")

    # if we're not going to force ourselves to re-analyze
    # events we've already analyzed, try and load existing
    # results so we know what we can skip.
    if not force:
        try:
            characterizations = pd.read_csv(
                results_dir / "characterizations.csv"
            ).set_index(["event_name", "norm_seconds"])
        except FileNotFoundError:
            characterizations = pd.DataFrame()

        try:
            timeseries = pd.read_csv(results_dir / "timeseries.csv")
            timeseries = timeseries.set_index(["event_name", "norm_seconds"])
        except FileNotFoundError:
            timeseries = pd.DataFrame()
    else:
        characterizations = timeseries = pd.DataFrame()

    # iterate through the segments and build a background
    # distribution on segments before known events
    thread_ex = AsyncExecutor(4, thread=True)
    process_ex = AsyncExecutor(4, thread=False)
    with thread_ex, process_ex:
        for i, segment in enumerate(timeslide.segments):
            if not any([t in segment for t in event_times]):
                continue

            # if this segment contains an event (or possibly multiple),
            # build up a background using as many earlier segments as
            # necessary to build up a background covering max_tb seconds
            # worth of data
            analyze_event(
                thread_ex,
                process_ex,
                characterizations=characterizations,
                timeseries=timeseries,
                event_segment=segment,
                background_segments=timeslide.segments[i - 1 :: -1],
                data_dir=data_dir,
                write_dir=write_dir,
                results_dir=results_dir,
                window_length=window_length,
                norm_seconds=norm_seconds,
                max_tb=max_tb,
                num_bins=num_bins,
                force=force,
            )


if __name__ == "__main__":
    main()
