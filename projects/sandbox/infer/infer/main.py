import logging
import time
from pathlib import Path
from typing import Callable, Iterator, List, Optional

import h5py
from infer.callback import Callback
from infer.data import ChunkedSegmentLoader, batch_chunks
from typeo import scriptify

from aframe.logging import configure_logging
from hermes.aeriel.client import InferenceClient


def infer_on_segment(
    client: InferenceClient,
    callback: Callable,
    sequence_id: int,
    it: Iterator,
    start: float,
    end: float,
    batch_size: int,
    inference_sampling_rate: float,
    sample_rate: float,
    throughput: float,
):
    str_rep = f"{int(start)}-{int(end)}"
    num_steps = callback.initialize(start, end)

    # create an iterator that will break
    # these chunks up into update-sized pieces
    batcher = batch_chunks(
        it,
        num_steps,
        batch_size,
        inference_sampling_rate,
        sample_rate,
        throughput,
    )

    duration = end - start
    logging.info(f"Beginning inference on {duration}s sequence {str_rep}")
    for i, (background, injected) in enumerate(batcher):
        if not (i + 1) % 10:
            logging.debug(f"Sending request {i + 1}/{num_steps}")

        client.infer(
            background,
            request_id=i,
            sequence_id=sequence_id,
            sequence_start=i == 0,
            sequence_end=i == (num_steps - 1),
        )
        client.infer(
            injected,
            request_id=i,
            sequence_id=sequence_id + 1,
            sequence_start=i == 0,
            sequence_end=i == (num_steps - 1),
        )

        # wait for the first response to come back
        # before proceeding in case the snapshot
        # state requires some warm up
        if not i:
            logging.debug("Waiting for initial response")
            callback.wait()

    # don't start inference on next sequence
    # until this one is complete
    result = client.get()
    while result is None:
        result = client.get()
        time.sleep(1e-1)

    logging.info(f"Retreived results from sequence {str_rep}")
    background_events, foreground_events = result
    return background_events, foreground_events


@scriptify
def main(
    ip: str,
    model_name: str,
    data_dir: Path,
    start: int,
    output_dir: Path,
    sample_rate: float,
    inference_sampling_rate: float,
    ifos: List[str],
    batch_size: int,
    integration_window_length: float,
    cluster_window_length: float,
    fduration: float,
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
    output_dir.mkdir(exist_ok=True, parents=True)
    configure_logging(log_file, verbose)

    callback = Callback(
        id=sequence_id,
        sample_rate=inference_sampling_rate,
        batch_size=batch_size,
        integration_window_length=integration_window_length,
        cluster_window_length=cluster_window_length,
        fduration=fduration,
    )

    # TODO: what's the best place to infer this info?
    # Obviously in the data loader, but how to communicate
    # to the batcher/callback?
    with h5py.File(data_dir / "background.hdf", "r") as f:
        size = len(f[ifos[0]][str(start)])
    duration = size / sample_rate
    end = start + duration

    logging.info(f"Connecting to server at {ip}:8001")
    client = InferenceClient(
        f"{ip}:8001", model_name, model_version, callback=callback
    )
    chunk_size = int(chunk_size * sample_rate)
    loader = ChunkedSegmentLoader(data_dir, str(start), ifos, chunk_size)
    with client, loader as loader:
        logging.info(f"Iterating through data from directory {data_dir}")
        background, foreground = infer_on_segment(
            client,
            callback,
            sequence_id=sequence_id,
            it=loader,
            start=start,
            end=end,
            batch_size=batch_size,
            inference_sampling_rate=inference_sampling_rate,
            sample_rate=sample_rate,
            throughput=throughput,
        )
        logging.info("Completed inference on all segments")

    logging.info("Building event sets and writing to files")
    background.write(output_dir / "background.hdf")
    foreground.write(output_dir / "foreground.hdf")


if __name__ == "__main__":
    main()
