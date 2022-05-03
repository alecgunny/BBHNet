import logging
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from hermes.typeo import typeo
from tqdm import tqdm

from bbhnet.analysis.distributions import DiscreteDistribution
from bbhnet.analysis.matched_filter import analyze_segment
from bbhnet.io.timeslides import Segment, TimeSlide
from bbhnet.logging import configure_logging
from bbhnet.parallelize import ProcessPool

event_times = [1186302519.8, 1186741861.5, 1187058327.1, 1187529256.5]
event_names = ["GW170809", "GW170814", "GW170818", "GW170823"]

pool = ProcessPool(2)
analyze_segment = pool.parallelize(analyze_segment)


def build_background(
    background_segments: Iterable[Segment],
    shifts: Iterable[str],
    write_dir: Path,
    max_tb: float,
    window_length: float = 1.0,
    norm_seconds: Optional[float] = None,
    num_bins: int = int(1e4),
):
    write_dir.mkdir(exist_ok=True)

    Tb = 0
    min_mf, max_mf = float("inf"), -float("inf")
    fnames = []
    for i, segment in enumerate(background_segments):
        segments = []
        for shift in shifts:
            try:
                shifted = segment.shift(shift)
            except ValueError:
                continue
            segments.append(shifted)

        logging.info(
            "Computing matched filter outputs on {} timeshifts "
            "of segment {}".format(len(segments), segment)
        )

        analyzer = analyze_segment(
            segments,
            window_length=window_length,
            norm_seconds=norm_seconds,
            write_dir=write_dir,
        )
        for fname, minval, maxval in tqdm(analyzer, total=len(segments)):
            fnames.append(fname)
            min_mf = min(min_mf, minval)
            max_mf = max(max_mf, maxval)

        Tb += len(shifts) * segment.length
        if Tb >= max_tb:
            logging.info(
                f"Accumulated {Tb}s of background matched filter outputs."
            )
            break

    logging.info(
        "Fitting discrete distribution of {} bins on range [{}, {}) "
        "to background measured using {} timeslides of segments:".format(
            num_bins, min_mf, max_mf, len(segments)
        )
    )
    for segment in background_segments[: i + 1]:
        logging.info(f"    {segment}")

    # TODO: best way to infer num bins?
    # TODO: should we parallelize the background fit?
    background = DiscreteDistribution("filtered", min_mf, max_mf, num_bins)
    background.fit(list(map(Segment, fnames)))

    logging.info(f"Background fit with {background.Tb}s worth of data")
    return background


def analyze_event(
    event_segment: Segment,
    event_time: float,
    background_segments: Iterable[Segment],
    shifts: Iterable[str],
    write_dir: Path,
    max_tb: float,
    window_length: float = 1.0,
    norm_seconds: Optional[float] = None,
    num_bins: int = int(1e4),
):
    # TODO: exclude segments with events?
    background = build_background(
        background_segments,
        shifts,
        write_dir,
        window_length=window_length,
        norm_seconds=norm_seconds,
        max_tb=max_tb,
        num_bins=num_bins,
    )

    # now use the fit background to characterize the
    # significance of BBHNet's detection around the event
    fname, _, __ = analyze_segment(
        event_segment,
        window_length=window_length,
        kernel_length=1,
        norm_seconds=norm_seconds,
        write_dir=write_dir,
    )

    far, t = background.characterize_events(Segment(fname), event_time)
    logging.info("False Alarm Rates: {}".format(list(far)))
    logging.info("Latencies: {}".format(list(t)))
    return far, t, background


def write_results(
    data: dict,
    norm_seconds: List[float],
    columns: List[str],
    fname: Path,
) -> None:
    event_names = list(data.keys())
    columns = pd.MultiIndex.from_product([event_names, norm_seconds, columns])
    values = np.stack([data[i][j][k] for i, j, k in columns.values]).T
    df = pd.DataFrame(values, columns=columns)
    df.to_csv(fname, index=False)


@typeo
def main(
    data_dir: Path,
    write_dir: Path,
    output_dir: Path,
    window_length: float = 1.0,
    norm_seconds: Optional[List[float]] = None,
    max_tb: Optional[float] = None,
    num_bins: int = 10000,
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
        output_dir: Path to directory to which to write analysis outputs
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
        log_file:
            A filename to write logs to. If left as `None`, logs will only
            be printed to stdout
        verbose:
            Flag indicating whether to log at level `INFO` (if set)
            or `DEBUG` (if not set)
    """

    configure_logging(output_dir / log_file, verbose)

    # organize timeslides into segments
    timeslide = TimeSlide(data_dir / "dt-0.0")
    shifts = [i for i in data_dir.iterdir() if i != "dt-0.0"]
    norm_seconds = norm_seconds or [norm_seconds]

    # iterate through the segments and build a background
    # distribution on segments before known events
    data = defaultdict(dict)
    for i, segment in enumerate(timeslide.segments):
        for event_time, event_name in zip(event_times, event_names):
            if event_time not in segment:
                continue

            # if this segment contains an event (or possibly multiple),
            # build up a background using as many earlier segments as
            # necessary to build up a background covering max_tb seconds
            # worth of data
            for norm in norm_seconds:
                logging.info(
                    "Computing false alarm rates and latencies for "
                    "event {} using a matched filter of length {}s "
                    "and normalization period {}s".format(
                        event_name, window_length, norm
                    )
                )
                far, t, background = analyze_event(
                    segment,
                    event_time,
                    timeslide.segments[i - 1 :: -1],
                    shifts,
                    write_dir,
                    window_length=window_length,
                    norm_seconds=norm,
                    max_tb=max_tb,
                    num_bins=num_bins,
                )

                results = {
                    "far": far,
                    "t": t,
                    "bin_centers": background.bin_centers,
                    "values": background.histogram,
                }
                data[event_name][norm] = results

    write_results(data, norm_seconds, ["far", "t"], output_dir / "far.csv")
    write_results(
        data,
        norm_seconds,
        ["values", "bin_centers"],
        output_dir / "background.csv",
    )
    return


if __name__ == "__main__":
    main()
