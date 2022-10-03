from functools import partial
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from bbhnet.io.h5 import write_timeseries
from bbhnet.io.timeslides import Segment


def replace_part(path: Path, part: str, field: str) -> Path:
    parts = path.parts
    idx = parts.index(part)
    p = Path(parts[0])
    parts = parts[1:idx] + (field,) + parts[idx + 1 :]
    return p.joinpath(*parts)


def find_shift_and_foreground(
    background_segment: Segment, shift: str, foreground_field: str
) -> Tuple[Optional[Segment], Optional[Segment]]:
    try:
        shifted = background_segment.make_shift(shift)
    except ValueError:
        return None, None

    replace = partial(replace_part, part=shifted.field, field=foreground_field)
    fnames = list(map(replace, shifted.fnames))
    try:
        injection = Segment(fnames)
    except ValueError:
        return shifted, None

    return shifted, injection


def load_segments(segments: Tuple[Segment]):
    background, foreground = segments
    yb, t = background.load("out")

    if foreground is not None:
        yf, _ = foreground.load("out")
    else:
        yf = None

    return yf, yb, t


def write_segment(
    write_dir: Path,
    shift: str,
    field: str,
    norm: Optional[float] = None,
    **fields: np.ndarray,
):
    if norm is not None:
        write_dir = write_dir / shift / f"{field}-norm-seconds.{norm}"
    else:
        write_dir = write_dir / shift / f"{field}"
    write_dir.mkdir(parents=True, exist_ok=True)

    return write_timeseries(write_dir, prefix="integrated", **fields)
