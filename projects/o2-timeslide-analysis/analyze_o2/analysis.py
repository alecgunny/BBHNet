from pathlib import Path
from typing import Optional

from analyze_o2 import utils
from hermes.typeo import typeo

from bbhnet.io import TimeSlide
from bbhnet.logging import configure_logging

event_times = [1186302519.8, 1186741861.5, 1187058327.1, 1187529256.5]
event_names = ["GW170809", "GW170814", "GW170818", "GW170823"]


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
    timeslide = TimeSlide(Path(data_dir) / "dt-0.0")

    # iterate through the segments and build a background
    # distribution on segments before known events
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

        # generate a background distribution using
        # timeshifts of the non-shifted segment
        background = utils.build_background(
            segment,
            write_dir,
            window_length=window_length,
            num_proc=8,
            norm_seconds=norm_seconds,
            max_tb=max_tb,
            num_bins=num_bins,
        )
        background.characterize_events(next_segment, event_time)


if __name__ == "__main__":
    main()
