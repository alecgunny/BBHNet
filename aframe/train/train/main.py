import os

import torch
from lightning.pytorch.cli import LightningCLI
from train.data import AframeDataset
from train.model import Aframe

from aframe.logging import configure_logging


class AframeCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments(
            "data.num_ifos",
            "model.arch.init_args.num_ifos",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data.valid_stride", "model.metric.init_args.stride"
        )


def main():
    cli = AframeCLI(
        model_class=Aframe,
        datamodule_class=AframeDataset,
        seed_everything_default=101588,
        run=False,
        parser_kwargs={"default_env": True},
        save_config_kwargs={"overwrite": True},
    )

    save_dir = cli.trainer.logger.save_dir
    if not save_dir.startswith("s3://"):
        os.makedirs(save_dir, exist_ok=True)
        log_file = os.path.join(save_dir, "train.log")
        configure_logging(log_file)
    else:
        configure_logging()
    cli.trainer.fit(cli.model, cli.datamodule)


if __name__ == "__main__":
    main()
