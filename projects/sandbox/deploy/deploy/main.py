import logging
from pathlib import Path
from typing import Callable, List

import numpy as np
import torch
from deploy.dataloading import data_iterator
from deploy.trigger import Searcher, Trigger

from bbhnet.architectures import Preprocessor, architecturize
from bbhnet.logging import configure_logging
from ml4gw.utils.slicing import unfold_windows


def integrate(x: torch.Tensor, w: torch.Tensor) -> np.ndarray:
    y = torch.nn.functional.conv1d(
        x[None, None], w[None, None], padding="valid"
    )
    return y[0, 0].cpu().numpy()


@architecturize
@torch.no_grad()
def main(
    architecture: Callable,
    outdir: Path,
    datadir: Path,
    ifos: List[str],
    channel: str,
    sample_rate: float,
    kernel_length: float,
    inference_sampling_rate: float,
    fduration: float,
    integration_window_length: float,
    refractory_period: float = 8,
    far_per_day: float = 1,
    verbose: bool = False,
):
    configure_logging(outdir / "log" / "deploy.log", verbose)

    kernel_size = int(kernel_length * sample_rate)
    stride = int(sample_rate / inference_sampling_rate)

    integrator_size = int(integration_window_length * inference_sampling_rate)
    integrator = torch.ones((integrator_size,)) / integrator_size
    integrator = integrator.to("cuda")

    taper = torch.hann_window(int(2 * sample_rate))[: int(sample_rate)]

    # instantiate network and preprocessor and
    # load in their optimized parameters
    weights_path = outdir / "training" / "weights.pt"
    logging.info(f"Build network and loading weights from {weights_path}")

    num_ifos = len(ifos)
    nn = architecture(num_ifos)
    preprocessor = Preprocessor(num_ifos, sample_rate, fduration=fduration)
    model = torch.nn.Sequential(preprocessor, nn).to("cuda")

    weights = torch.load(weights_path)
    model.load_state_dict(weights)
    model.eval()

    # set up some objects to use for finding
    # and submitting triggers
    searcher = Searcher(
        outdir,
        far_per_day,
        inference_sampling_rate,
        refractory_period,
        offset=kernel_length - integration_window_length - fduration,
    )

    trigger_dir = outdir / "triggers"
    trigger_dir.mkdir(exist_ok=True)
    trigger = Trigger(trigger_dir)

    logging.info("Beginning search")
    data_it = data_iterator(datadir, channel, ifos, sample_rate, timeout=10)
    in_spec = False
    integrated = None  # need this for static linters
    for X, t0, ready in data_it:
        if not ready:
            # check if this is because the frame stream stopped
            # being analysis ready, or if it's because frames
            # were dropped within the stream
            if X is not None:
                logging.warning(f"Frame {t0} not analysis ready, skipping")
            else:
                logging.warning(
                    "Missing frame files after timestep {}, "
                    "resetting states".format(t0)
                )

            # if we had an event in the last frame, we
            # won't get to see its peak, so do our best
            # to build the event with what we have
            if searcher.detecting:
                event = searcher.build_event(
                    integrated[-1], t0 - 1, len(integrated) - 1
                )
                trigger.submit(event, ifos)
                searcher.detecting = False

            in_spec = False
            continue
        elif not in_spec:
            # the frame is analysis ready, but previous frames
            # weren't, so reset our running states and taper
            # the data in to not add frequency artifacts
            logging.info(f"Frame {t0} is ready again, resetting states")
            window = torch.zeros((2, kernel_size - stride), device="cuda")
            outputs = torch.zeros((integrator_size - 1,), device="cuda")
            X *= taper
            in_spec = True

        X = X.to("cuda")
        window = torch.cat([window, X], axis=-1)

        # TODO: this won't generalize to stride + kernel_size
        # combinations that don't neatly fit into the window,
        # but whatever...
        batch = unfold_windows(window, kernel_size, stride)
        y = model(batch)[:, 0]
        outputs = torch.cat([outputs, y])
        integrated = integrate(outputs, integrator)

        # slough off old input and output data
        window = window[:, X.shape[-1] :]
        outputs = outputs[integrator_size - 1 :]

        event = searcher.search(integrated, t0)
        if event is not None:
            trigger.submit(event, ifos)
