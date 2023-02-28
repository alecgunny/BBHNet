import time
from pathlib import Path
from typing import List, Optional

from infer.callback import Callback
from infer.data import load_segments
from infer.sequence import Sequence
from typeo import scriptify

from bbhnet.analysis.events import (
    EventSet,
    RecoveredInjectionSet,
    TimeSlideEventSet,
)
from bbhnet.logging import configure_logging
from hermes.aeriel.client import InferenceClient


@scriptify
def main(
    ip: str,
    model_name: str,
    data_dir: Path,
    output_fname: Path,
    injection_set_file: Path,
    sample_rate: float,
    inference_sampling_rate: float,
    shifts: List[float],
    ifos: List[str],
    batch_size: int,
    integration_window_length: float,
    cluster_window_length: float,
    throughput: float,
    chunk_size: float,
    sequence_id: int,
    model_version: int = -1,
    log_file: Optional[Path] = None,
    verbose: bool = False,
) -> None:
    """
    Perform inference using Triton on a directory
    of timeseries files, using a particular set of
    interferometer time shifts. Network outputs will
    be saved both as-is and using local integration.

    Args:
        ip:
            The IP address at which a Triton server
            hosting the indicated model is running
        model_name:
            The name of the model to which to make
            inference requests
        data_dir:
            Directory containing input files representing
            timeseries on which to perform inference.
            Each HDF5 file in this directory will be used
            for inference.
        write_dir:
            Directory to which to save raw and locally
            integrated network outputs.
        sample_rate:
            Rate at which input timeseries data has
            been sampled.
        inference_sampling_rate:
            The rate at which to sample windows for
            inference from the input timeseries.
            Corresponds to the sample rate of the
            output timeseries.
        batch_size:
            The number of subsequent windows to
            include in a single batch of inference.
        window_length:
            Length of the window over which network
            outputs should be locally integrated,
            specified in seconds.
        max_shift:
            The maximum shift value across all runs
            in the timeslide analysis that this run
            is a part of. This helps keep all output
            timeseries the same length.
        throughput:
            Rate at which to make requests, in units
            of seconds of data per second `[s' / s]`.
        sequence_id:
            Identifier to assign to all the sequences
            of inference requests in this run to match up
            with a corresponding snapshot state on the
            inference server.
        model_version:
            Version of the model from which to request
            inference. Default value of `-1` indicates
            the latest available version of the model.
        log_file:
            File to which to write inference logs.
        verbose:
            Flag controlling whether logging verbosity
            is `DEBUG` (`True`) or `INFO` (`False`)
    """
    configure_logging(log_file, verbose)

    callback = Callback(
        integration_window_length=integration_window_length,
        cluster_window_length=cluster_window_length,
    )
    client = InferenceClient(
        f"{ip}:8001", model_name, model_version, callback=callback
    )
    loader = load_segments(
        data_dir,
        ifos=ifos,
        chunk_size=chunk_size,
        shifts=shifts,
        sample_rate=sample_rate,
    )
    with client:
        background_events = TimeSlideEventSet()
        foreground_events = RecoveredInjectionSet()
        for (start, stop), it in loader:
            sequence = Sequence(
                start,
                stop,
                inference_sampling_rate,
                batch_size,
                injection_set_file,
            )
            callback.register(sequence)
            sequence_it = sequence.iter(it, ifos, sample_rate, throughput)
            for i, (background, injected) in enumerate(sequence_it):
                client.infer(
                    background,
                    request_id=i,
                    sequence_id=sequence_id,
                    sequence_start=i == 0,
                    sequence_end=i == (len(sequence) - 1),
                )
                client.infer(
                    injected,
                    request_id=i,
                    sequence_id=sequence_id + 1,
                    sequence_start=i == 0,
                    sequence_end=i == (len(sequence) - 1),
                )

            # don't start inference on next sequence
            # until this one is complete
            while True:
                result = client.get()
                if result is not None:
                    bckgrd_events, frgrd_events = result
                    background_events.append(bckgrd_events)
                    foreground_events.append(frgrd_events)
                    break
                time.sleep(1e-1)

    background_events = EventSet.from_timeslide(background_events, shifts)
    background_events.write(output_fname)
    foreground_events.write(output_fname)


if __name__ == "__main__":
    main()
