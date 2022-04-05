import logging
import sys
from pathlib import Path
from typing import Optional

from analyze_o2.utils import build_background
from hermes.typeo import typeo

from bbhnet.io import TimeSlide

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
    logging.basicConfig(
        stream=sys.stdout,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG if verbose else logging.INFO,
    )
    if log_file is not None:
        handler = logging.FileHandler(filename=log_file, mode="w")
        logging.getLogger().addHandler(handler)

    timeslide = TimeSlide(Path(data_dir) / "dt-0.0")
    for segment, next_segment in zip(
        timeslide.segments[:-1], timeslide.segments[1:]
    ):
        for event_time, event_name in zip(event_times, event_names):
            if event_time in next_segment:
                analyzed_fnames, Tb, min_value, max_value = build_background(
                    data_dir,
                    write_dir,
                    num_bins,
                    window_length=window_length,
                    fnames=segment.fnames,
                    num_proc=8,
                    norm_seconds=norm_seconds,
                    max_tb=max_tb,
                )
                # do analysis of segment here
                break


if __name__ == "__main__":
    main()
