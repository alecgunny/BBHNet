from pathlib import Path
from typing import List, Tuple

import numpy as np
from typeo import scriptify

from aframe.utils.datagen import calc_shifts_required, make_fname


def get_num_shifts(
    segments: List[Tuple[float, float]], Tb: float, shift: float
) -> int:
    """
    Calculates the number of required time shifts based on a list
    of background segments
    """
    T = sum([stop - start for start, stop in segments])
    return calc_shifts_required(Tb, T, shift)


@scriptify
def main(
    segments: Path,
    Tb: float,
    shifts: List[float],
    psd_length: float,
    background_dir: Path,
):
    segments = np.loadtxt(segments)
    shifts_required = get_num_shifts(segments, Tb, max(shifts))
    # create text file from which the condor job will read
    # the start, stop, and shift for each job
    parameters = "start,stop,shift,psd_file\n"
    for start, stop in segments:
        psd_file = make_fname("background", start, stop)
        psd_file = background_dir / psd_file
        for i in range(shifts_required):
            # TODO: make this more general
            shift = [(i + 1) * shift for shift in shifts]
            shift = " ".join(map(str, shift))
            # add psd_length to account for the burn in of psd calculation
            parameters += f"{start + psd_length},{stop},{shift}\n"


if __name__ == "__main__":
    main()
