from pathlib import Path
from typing import List

from typeo import scriptify

from mldatafind.authenticate import authenticate
from mldatafind.io import fetch_timeseries


@scriptify
def main(
    start: float,
    stop: float,
    channel: str,
    ifos: List[str],
    sample_rate: float,
    write_path: Path,
):
    """Generates background data for training and testing aframe

    Args:
        start:
            Starting GPS time of the timeseries to be fetched
        stop:
            Ending GPS time of the timeseries to be fetched
        writepath:
            Path, including file name, that the data will be saved to
        channel:
            Channel from which to fetch the timeseries
        ifos:
            List of interferometers to query data from. Expected to be given
            by prefix; e.g. "H1" for Hanford
        sample_rate:
            Sample rate to which the timeseries will be resampled, specified
            in Hz

    Returns: The `Path` of the output file
    """

    authenticate()
    channels = [f"{ifo}:{channel}" for ifo in ifos]
    data = fetch_timeseries(
        channels, start, stop, allow_tape=True, verbose=True
    )
    data = data.resample(sample_rate)
    for ifo in ifos:
        data[ifo] = data.pop(f"{ifo}:{channel}")

    data.write(write_path)
    return write_path


if __name__ == "__main__":
    main()
