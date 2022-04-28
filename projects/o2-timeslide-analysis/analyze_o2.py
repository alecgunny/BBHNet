import logging
from pathlib import Path
from typing import Optional

import numpy as np
from hermes.typeo import typeo
from tqdm import tqdm

from bbhnet.analysis.distributions import DiscreteDistribution
from bbhnet.analysis.matched_filter import analyze_segments_parallel
from bbhnet.io.timeslides import Segment, TimeSlide
from bbhnet.logging import configure_logging

event_times = [1186302519.8, 1186741861.5, 1187058327.1, 1187529256.5]
event_names = ["GW170809", "GW170814", "GW170818", "GW170823"]


def build_background(
    segment: Segment,
    write_dir: str,
    window_length: float = 1.0,
    num_proc: int = 1,
    norm_seconds: Optional[float] = None,
    max_tb: Optional[float] = None,
    num_bins: int = int(1e4),
):
    # TODO: technically doesn't cover the case where
    # segment.length is a multiple but I mean... what
    # are the chances
    segments = [segment]
    extra_segments = range(1, int(max_tb // segment.length) + 1)
    segments += [segment.shift(f"dt-{i * 0.1}") for i in extra_segments]
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

    logging.info(
        "Fitting discrete distribution of {} bins on range [{}, {}) "
        "to background on measured on segment {} using {} files".format(
            num_bins, min_mf, max_mf, segment, len(fnames)
        )
    )
    # TODO: best way to infer num bins?
    # TODO: should we parallelize the background fit?
    background = DiscreteDistribution(min_mf, max_mf, num_bins)
    background.fit(list(map(Segment, fnames)))

    logging.info(
        "Background fit with {} seconds worth of data".format(background.Tb)
    )
    return background


@typeo
def main(
    data_dir: Path,
    write_dir: Path,
    analysis_period: float,
    num_bins: int = 10000,
    window_length: float = 1.0,
    norm_seconds: Optional[float] = None,
    max_tb: Optional[float] = None,
    log_file: Optional[str] = None,
    verbose: bool = False,
):
    configure_logging(log_file, verbose)

    # organize timeslides into segments
    timeslide = TimeSlide(data_dir / "dt-0.0")

    # iterate through the segments and build a background
    # distribution on segments before known events
    fars, ts = [], []
    for segment, next_segment in zip(
        timeslide.segments[:-1], timeslide.segments[1:]
    ):
        # check if any of the events fall in the (i+1)th segment
        for event_time, event_name in zip(event_times, event_names):
            if event_time in next_segment:
                break
        else:
            # the loop never broke, so there's no event
            # to build background for, so move on
            continue

        logging.info(
            "Building background on segment {} to characterize "
            "event {} at time {} in segment {}".format(
                segment, event_name, event_time, next_segment
            )
        )

        # generate a background distribution using
        # timeshifts of the non-shifted segment
        background = build_background(
            segment,
            write_dir,
            window_length=window_length,
            num_proc=8,
            norm_seconds=norm_seconds,
            max_tb=max_tb,
            num_bins=num_bins,
        )

        # now use the fit background to characterize
        # the significance of BBHNet's detection around
        # the event
        far, t = background.characterize_events(next_segment, event_time)
        logging.info("False Alarm Rates: {}".format(list(far)))
        logging.info("Latencies: {}".format(list(t)))

        fars.append(far)
        ts.append(t)

    return np.stack(fars), np.stack(ts)


if __name__ == "__main__":
    main()
