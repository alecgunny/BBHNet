import os
import yaml
from math import ceil
from tempfile import NamedTemporaryFile
from typing import Optional

import ray
from ray import tune
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from ray.train.torch import TorchTrainer
from ray.tune.schedulers import ASHAScheduler
from train.cli import AframeCLI

from aframe.logging import configure_logging

search_space = {
    "model.learning_rate": tune.loguniform(1e-4, 1e-1),
    "data.swap_frac": tune.uniform(0, 0.2),
    "data.mute_frac": tune.uniform(0, 0.2)
}


class TuneClientCLI(AframeCLI):
    def __init__(self, *args, search_space: dict, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in search_space.items():
            # TODO: potentially strip these out from
            # the parser so that we can specify configs
            # that don't specify them
            pass

    # since this is run on the client, we don't actually
    # want to do anything with the arguments we parse,
    # just record them, so override the couple parent
    # methods responsible for actually doing stuff
    def instantiate_classes(self):
        return

    def _run_subcommand(self):
        return

    def add_arguments_to_parser(self, parser):
        """
        Add some arguments about where and how to
        run the tune job.
        """
        super().add_arguments_to_parser(parser)
        parser.add_argument("--tune.name", type=str, default="ray-tune")
        parser.add_argument("--tune.address", type=str, default=None)
        parser.add_argument("--tune.gpus_per_job", type=int, default=1)
        parser.add_argument("--tune.cpus_per_job", type=int, default=8)
        parser.add_argument("--tune.num_samples", type=int, default=10)

        # this argument isn't valuable for that much, but when
        # we try to deploy on local containers on LDG, the default
        # behavior will be to make a temp directory for ray cluster
        # logs at /local, which will cause permissions issues.
        parser.add_argument("--tune.temp_dir", type=str, default=None)


class TuneCLI(AframeCLI):
    """
    CLI to use in the actual training jobs themselves,
    adds in some of the Ray magic we need to make things work
    """
    def instantiate_trainer(self, **kwargs):
        kwargs = kwargs | dict(
            devices="auto",
            accelerator="auto",
            strategy=RayDDPStrategy(),
            callbacks=[RayTrainReportCallback()],
            plugins=[RayLightningEnvironment()],
            enable_progress_bar=False
        )
        return super().instantiate_trainer(**kwargs)


class TrainFunc:
    def __init__(self, name: str, config: dict) -> None:
        self.name = name
        self.config = config

    def __call__(self, config):
        with NamedTemporaryFile(mode="w") as f:
            yaml.dump(self.config, f)
            args = ["-c", f.name]
            for key, value in config.items():
                args.append(f"--{key}={value}")

            # TODO: this is technically W&B specific,
            # but if we're distributed tuning I don't
            # really know what other logger we would use
            args.append(f"--trainer.logger.group={self.name}")
            cli = TuneCLI(
                run=False,
                parser_kwargs={"default_env": True},
                save_config_kwargs={"overwrite": True},
                args=args
            )

        save_dir = cli.trainer.logger.save_dir
        if not save_dir.startswith("s3://"):
            os.makedirs(save_dir, exist_ok=True)
            log_file = os.path.join(save_dir, "train.log")
            configure_logging(log_file)
        else:
            configure_logging()

        trainer = prepare_trainer(cli.trainer)
        trainer.fit(cli.model, cli.datamodule)


def configure_deployment(
    train_func: TrainFunc,
    gpus_per_job: int,
    cpus_per_job: int
) -> TorchTrainer:
    cpus_per_worker = ceil(cpus_per_job / gpus_per_job)
    scaling_config = ScalingConfig(
        num_workers=gpus_per_job,
        use_gpu=True,
        resources_per_worker={"CPU": cpus_per_worker, "GPU": 1}
    )
    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute="valid_auroc",
            checkpoint_score_order="max",
        ),
    )
    return TorchTrainer(
        train_func,
        scaling_config=scaling_config,
        run_config=run_config,
    )


def main(args: Optional[list[str]] = None):
    # create a yaml dict version of whatever arguments
    # we passed at the command line to pass again in
    # each train  job
    cli = TuneClientCLI(
        run=False,
        parser_kwargs={"default_env": True},
        save_config_kwargs={"overwrite": True},
        search_space=search_space,
        args=args
    )
    config = cli.parser.dump(cli.config, format="yaml")
    config = yaml.safe_load(config)

    # pop out the arguments specific to the tuning
    # and initialize a session if there's any existing
    # cluster we should connect to
    tune_config = config.pop("tune")
    if "address"  in tune_config:
        address = "ray://" + tune_config.pop("address")
    else:
        address = None
    ray.init(address, _temp_dir=tune_config["temp_dir"])

    # construct the function that will actually
    # execute the training loop, and then set it
    # up for Ray to distribute it over our cluster,
    # with the desired number of resources allocated
    # to each running version of the job
    train_func = TrainFunc(tune_config["name"], config)
    train_func = configure_deployment(
        train_func,
        tune_config["gpus_per_job"],
        tune_config["cpus_per_job"]
    )
    scheduler = ASHAScheduler(max_t=2, grace_period=1, reduction_factor=2)
    tuner = tune.Tuner(
        train_func,
        param_space={"train_loop_config": search_space},
        tune_config=tune.TuneConfig(
            metric="valid_auroc",
            mode="max",
            num_samples=tune_config["num_samples"],
            scheduler=scheduler,
        ),
    )
    return tuner.fit()


if __name__ == "__main__":
    main()
