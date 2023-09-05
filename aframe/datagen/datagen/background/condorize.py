from pathlib import Path

# from aframe.datagen import make_fname
import numpy as np
from typeo import scriptify


def _intify(x: float):
    """
    Converts the input float into an int if the two are equal (e.g., 4.0 == 4).
    Otherwise, returns the input unchanged.
    """
    return int(x) if int(x) == x else x


def make_fname(prefix, t0, length):
    """Creates a filename for background files in a consistent format"""
    t0 = _intify(t0)
    length = _intify(length)
    return f"{prefix}-{t0}-{length}.hdf5"


@scriptify
def main(
    segment_path: Path,
    background_dir: Path,
    output_file: Path,
):
    segments = np.loadtxt(segment_path)
    # determine which segments need to be generated
    parameters = []
    for start, stop in segments:
        duration = stop - start
        fname = make_fname("background", start, duration)
        write_path = background_dir / fname

        if not write_path.exists():
            parameters.append([start, stop, write_path])

    parameters = np.vstack(parameters)

    with open(output_file, "w") as f:
        f.write("start,stop,writepath\n")
        for start, stop, writepath in parameters:
            f.write(f"{start},{stop},{writepath}\n")

    return output_file


if __name__ == "__main__":
    main()
