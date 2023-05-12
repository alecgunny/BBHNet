import logging
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
from deploy.dataloading import data_iterator

from bbhnet.analysis.ledger.events import EventSet
from bbhnet.architectures import ResNet
from bbhnet.logging import configure_logger


def unfold(x, kernel_size: int, stride: int):
    # TODO: fill out
    return


def submit_trigger():
    # TODO: fill out
    return


def main(
    outdir: Path,
    datadir: Path,
    ifos: List[str],
    channel: str,
    sample_rate: float,
    kernel_length: float,
    inference_sampling_rate: float,
    integration_window_length: float,
    refractory_period: float = 8,
    far_per_day: float = 1,
    verbose: bool = False,
):
    configure_logger(outdir / "log" / "deploy.log", verbose)

    logging.debug("Loading background measurements")
    background = EventSet.read(outdir / "infer" / "background.h5")
    num_events = int(far_per_day * background.Tb / 3600 / 24)
    if not num_events:
        raise ValueError(
            "Background livetime {}s not enough to detect "
            "events with false alarm rate of {} per day".format(
                background.Tb, far_per_day
            )
        )
    threshold = np.sort(background.detection)[-num_events]
    logging.info(
        "FAR {}/day threshold is {:0.3f}".format(far_per_day, threshold)
    )

    kernel_size = int(kernel_length * sample_rate)
    stride = int(sample_rate / inference_sampling_rate)

    integrator_size = int(integration_window_length * sample_rate)
    integrator = torch.ones((integrator_size,)) / integrator_size

    # TODO: include preprocessor
    weights_path = outdir / "training" / "weights.pt"
    logging.info(f"Build network and loading weights from {weights_path}")
    nn = ResNet(2).to("cuda")
    nn.load_state_dict(torch.load(weights_path))

    # initialize input and output states to zeros
    window = torch.zeros((2, kernel_size), device="cuda")
    outputs = torch.zeros((integrator_size - 1,), device="cuda")

    # set up some parameters to use for figuring
    # out if/when a trigger happens
    detecting = False
    last_detection_time = time.time()

    logging.info("Beginning search")
    data_it = data_iterator(datadir, channel, ifos, sample_rate, timeout=5)
    for X in data_it:
        X = X.to("cuda")

        # TODO: taper first X
        window = torch.cat([window, X], axis=-1)

        # TODO: grab unfolding code from somewhere
        batch = unfold(window, kernel_size, stride)
        y = nn(batch)[:, 0]
        outputs = torch.cat([outputs, y])

        # TODO: need to add the right dimensions here
        integrated = (
            torch.nn.functional.conv1d(outputs, integrator, mode="valid")
            .cpu()
            .numpy()
        )

        # slough off old input and output data
        window = window[:, X.shape[-1] :]
        outputs = outputs[integrator_size - 1 :]

        time_since_last = time.time() - last_detection_time
        if not detecting:
            if (integrated >= threshold).any():
                if time_since_last < refractory_period:
                    logging.warning(
                        "Detected event with detection statistic {:0.3f} "
                        "but it has only been {}s since last detection, "
                        "so skipping".format(integrated.max(), time_since_last)
                    )
                    continue

                logging.info(
                    "Detected event with detection statistic {:0.3f}".format(
                        integrated.max()
                    )
                )
                idx = np.argmax(integrated)
                if idx < (len(integrated) - 1):
                    # TODO: what else do we need here?
                    # need to keep track of timestamps for sure
                    logging.info("Event time found to be ")
                    submit_trigger()
                    last_detection_time = time.time()
                else:
                    detecting = True
        elif detecting:
            logging.info("Event time found to be ")
            idx = np.argmax(integrated)
            submit_trigger()
            detecting = False
            last_detection_time = time.time()
