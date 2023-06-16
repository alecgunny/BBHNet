import logging
from pathlib import Path
from typing import Callable, Optional

import torch
from export.snapshotter import add_streaming_input_preprocessor

import hermes.quiver as qv
from aframe.architectures import architecturize
from aframe.logging import configure_logging


def scale_model(model, instances):
    # TODO: should quiver handle this under the hood?
    try:
        model.config.scale_instance_group(instances)
    except ValueError:
        model.config.add_instance_group(count=instances)


@architecturize
def main(
    architecture: Callable,
    repository_directory: str,
    outdir: Path,
    num_ifos: int,
    kernel_length: float,
    inference_sampling_rate: float,
    sample_rate: float,
    batch_size: int,
    fduration: float,
    background_length: float = 16,
    fftlength: float = 8,
    highpass: Optional[float] = None,
    weights: Optional[Path] = None,
    streams_per_gpu: int = 1,
    aframe_instances: Optional[int] = None,
    platform: qv.Platform = qv.Platform.ONNX,
    clean: bool = False,
    verbose: bool = False,
) -> None:
    """
    Export a aframe architecture to a model repository
    for streaming inference, including adding a model
    for caching input snapshot state on the server.

    Args:
        architecture:
            A function which takes as input a number of witness
            channels and returns an instantiated torch `Module`
            which represents a DeepClean network architecture
        repository_directory:
            Directory to which to save the models and their
            configs
        outdir:
            Path to save logs. If `weights` is `None`, this
            directory is assumed to contain a file `"weights.pt"`.
        num_ifos:
            The number of interferometers contained along the
            channel dimension used to train aframe
        inference_sampling_rate:
            The rate at which kernels are sampled from the
            h(t) timeseries. This, along with the `sample_rate`,
            dictates the size of the update expected at the
            snapshotter model
        sample_rate:
            Rate at which the input kernel has been sampled, in Hz
        weights:
            Path to a set of trained weights with which to
            initialize the network architecture. If left as
            `None`, a file called `"weights.pt"` will be looked
            for in the `output_directory`.
        streams_per_gpu:
            The number of snapshot states to host per GPU during
            inference
        instances:
            The number of concurrent execution instances of the
            aframe architecture to host per GPU during inference
        platform:
            The backend framework platform used to host the
            DeepClean architecture on the inference service. Right
            now only `"onnxruntime_onnx"` is supported.
        clean:
            Whether to clear the repository directory before starting
            export
        verbose:
            If set, log at `DEBUG` verbosity, otherwise log at
            `INFO` verbosity.
        **kwargs:
            key word arguments specific to the export platform
    """

    # make relevant directories
    outdir.mkdir(exist_ok=True, parents=True)

    # if we didn't specify a weights filename, assume
    # that a "weights.pt" lives in our output directory
    if weights is None or weights.is_dir():
        weights_dir = outdir if weights is None else weights
        weights = weights_dir / "weights.pt"
    if not weights.exists():
        raise FileNotFoundError(f"No weights file '{weights}'")

    configure_logging(outdir / "export.log", verbose)

    # instantiate a new, randomly initialized version
    # of the network architecture, including preprocessor
    logging.info("Initializing model architecture")
    nn = architecture(num_ifos)
    logging.info(f"Initialize:\n{nn}")

    # load in a set of trained weights
    logging.info(f"Loading parameters from {weights}")
    state_dict = torch.load(weights, map_location="cpu")
    nn.load_state_dict(state_dict)
    nn.eval()

    # instantiate a model repository at the
    # indicated location. Split up the preprocessor
    # and the neural network (which we'll call aframe)
    # to export/scale them separately, and start by
    # seeing if either already exists in the model repo
    repo = qv.ModelRepository(repository_directory, clean)
    try:
        aframe = repo.models["aframe"]
    except KeyError:
        aframe = repo.add("aframe", platform=platform)

    # if we specified a number of instances we want per-gpu
    # for each model at inference time, scale them now
    if aframe_instances is not None:
        scale_model(aframe, aframe_instances)

    size = int((kernel_length - fduration) * sample_rate)
    input_shape = (batch_size, num_ifos, size)
    # the network will have some different keyword
    # arguments required for export depending on
    # the target inference platform
    # TODO: hardcoding these kwargs for now, but worth
    # thinking about a more robust way to handle this
    kwargs = {}
    if platform == qv.Platform.ONNX:
        kwargs["opset_version"] = 13

        # turn off graph optimization because of this error
        # https://github.com/triton-inference-server/server/issues/3418
        aframe.config.optimization.graph.level = -1
    elif platform == qv.Platform.TENSORRT:
        kwargs["use_fp16"] = False

    aframe.export_version(
        nn,
        input_shapes={"whitened": input_shape},
        output_names=["discriminator"],
        **kwargs,
    )

    # now try to create an ensemble that has a snapshotter
    # at the front for streaming new data to
    ensemble_name = "aframe-stream"
    try:
        # first see if we have an existing
        # ensemble with the given name
        ensemble = repo.models[ensemble_name]
    except KeyError:
        # if we don't, create one
        ensemble = repo.add(ensemble_name, platform=qv.Platform.ENSEMBLE)
        whitened = add_streaming_input_preprocessor(
            ensemble,
            aframe.inputs["whitened"],
            background_length=background_length,
            sample_rate=sample_rate,
            inference_sampling_rate=inference_sampling_rate,
            fduration=fduration,
            fftlength=fftlength,
            highpass=highpass,
            name="snapshotter",
            streams_per_gpu=streams_per_gpu,
        )
        ensemble.pipe(whitened, aframe.inputs["whitened"])

        # export the ensemble model, which basically amounts
        # to writing its config and creating an empty version entry
        ensemble.add_output(aframe.outputs["discriminator"])
        ensemble.export_version(None)
    else:
        # if there does already exist an ensemble by
        # the given name, make sure it has aframe
        # and the snapshotter as a part of its models
        if aframe not in ensemble.models:
            raise ValueError(
                "Ensemble model '{}' already in repository "
                "but doesn't include model 'aframe'".format(ensemble_name)
            )
        # TODO: checks for snapshotter and preprocessor

    # keep snapshot states around for a long time in case there are
    # unexpected bottlenecks which throttle update for a few seconds
    snapshotter = repo.models["snapshotter"]
    snapshotter.config.sequence_batching.max_sequence_idle_microseconds = int(
        6e10
    )
    snapshotter.config.write()


if __name__ == "__main__":
    main()
