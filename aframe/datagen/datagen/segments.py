from pathlib import Path
from typing import List

import numpy as np
from typeo import scriptify

from mldatafind.segments import query_segments


@scriptify
def main(
    start: float,
    stop: float,
    state_flag: str,
    minimum_length: float,
    ifos: List[str],
    output_file: Path,
):
    """Wrapper around query_segments to write segments
    to a file for compatibility with Luigi Tasks
    """

    segments = query_segments(
        [f"{ifo}:{state_flag}" for ifo in ifos],
        start,
        stop,
        minimum_length,
    )

    np.savetxt(
        output_file,
        segments,
    )
    return output_file


if __name__ == "__main__":
    main()
