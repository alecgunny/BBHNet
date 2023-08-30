import logging
from pathlib import Path
from typing import Callable

import h5py
from typeo import scriptify

from aframe.injection import generate_gw
from aframe.logging import configure_logging


@scriptify
def main(
    prior: Callable,
    num_signals: int,
    output_file: Path,
    log_file: Path,
    reference_frequency: float,
    minimum_frequency: float,
    sample_rate: float,
    waveform_duration: float,
    waveform_approximant: str = "IMRPhenomPv2",
    verbose: bool = False,
):
    """
    Simulates a set of BBH plus and cross polarization waveforms
    and saves them to an output file.

    Args:
        prior:
            A function that returns a Bilby PriorDict when called
        num_signals:
            Number of waveforms to simulate
        datadir:
            Directory to which the waveforms will be written
        logdir:
            Directory to which the log file will be written
        reference_frequency:
            Frequency of the gravitational wave at the state of
            the merger that other quantities are defined with
            reference to
        minimum_frequency:
            Minimum frequency of the gravitational wave. The part
            of the gravitational wave at lower frequencies will
            not be generated. Specified in Hz.
        sample_rate:
            Sample rate at which the waveforms will be simulated,
            specified in Hz
        waveform_duration:
            Length of the waveforms in seconds
        waveform_approximant:
            The lalsimulation waveform approximant to use
        force_generation:
            If False, will not generate data if an existing dataset exists
        verbose:
            If True, log at `DEBUG` verbosity, otherwise log at
            `INFO` verbosity.

    Returns: The name of the file containing the waveforms and parameters
    """

    configure_logging(log_file, verbose)

    # log and print out some simulation parameters
    logging.info("Simulation parameters")
    logging.info("Number of samples     : {}".format(num_signals))
    logging.info("Sample rate [Hz]      : {}".format(sample_rate))
    logging.info("Prior name            : {}".format(prior.__name__))

    # sample gw parameters
    prior, detector_frame_prior = prior()
    params = prior.sample(num_signals)

    signals = generate_gw(
        params,
        minimum_frequency,
        reference_frequency,
        sample_rate,
        waveform_duration,
        waveform_approximant,
        detector_frame_prior,
    )

    with h5py.File(output_file, "w") as f:
        # write signals attributes, snr, and signal parameters
        for k, v in params.items():
            f.create_dataset(k, data=v)

        f.create_dataset("signals", data=signals)

        # write attributes
        f.attrs.update(
            {
                "size": num_signals,
                "sample_rate": sample_rate,
                "waveform_duration": waveform_duration,
                "waveform_approximant": waveform_approximant,
                "reference_frequency:": reference_frequency,
                "minimum_frequency": minimum_frequency,
            }
        )
    return output_file


if __name__ == "__main__":
    main()
