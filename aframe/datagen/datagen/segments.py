import logging
from pathlib import Path
from typing import List

import numpy as np
from typeo import scriptify

from mldatafind.authenticate import authenticate
from mldatafind.segments import query_segments


def split_segments(segments: List[tuple], chunk_size: float):
    """
    Split a list of segments into segments that are at most
    `chunk_size` seconds long
    """
    out_segments = []
    for segment in segments:
        start, stop = segment
        duration = stop - start
        if duration > chunk_size:
            num_segments = int((duration - 1) // chunk_size) + 1
            logging.info(f"Chunking segment into {num_segments} parts")
            for i in range(num_segments):
                end = min(start + (i + 1) * chunk_size, stop)
                seg = (start + i * chunk_size, end)
                out_segments.append(seg)
        else:
            out_segments.append(segment)
    return out_segments


@scriptify
def main(
    start: float,
    stop: float,
    state_flag: str,
    minimum_length: float,
    maximum_length: float,
    ifos: List[str],
    output_file: Path,
):
    """Wrapper around query_segments to write segments
    to a file for compatibility with Luigi Tasks
    """
    authenticate()
    segments = query_segments(
        [f"{ifo}:{state_flag}" for ifo in ifos],
        start,
        stop,
        minimum_length,
    )
    segments = split_segments(segments, maximum_length)

    np.savetxt(
        output_file,
        segments,
    )
    return output_file


if __name__ == "__main__":
    main()
