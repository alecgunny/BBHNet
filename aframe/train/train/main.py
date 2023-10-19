import os

import torch
from lightning.pytorch.cli import LightningCLI
from train.data import AframeDataset
from train.model import Aframe

from aframe.logging import configure_logging


class AframeCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_optimizer_args(torch.optim.Adam)
        parser.add_lr_scheduler_args(torch.optim.lr_scheduler.OneCycleLR)

        parser.link_arguments(
            "data.num_ifos",
            "model.arch.init_args.num_ifos",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data.num_ifos",
            "model.arch.init_args.groups",
            apply_on="instantiate"
        )
        parser.link_arguments(
            "data.num_ifos",
            "model.arch.init_args.skip_connection.init_args.groups",
            apply_on="instantiate"
        )

        parser.link_arguments(
            "data.sample_rate",
            "model.loss_fn.init_args.sample_rate",
            apply_on="instantiate"
        )
        parser.link_arguments(
            "data.kernel_length",
            "model.loss_fn.init_args.chisq_kernel_length",
            apply_on="parse"
        )
        parser.link_arguments(
            "data.highpass",
            "model.loss_fn.init_args.chisq_highpass",
            apply_on="parse"
        )

        parser.link_arguments(
            "data.steps_per_epoch",
            "lr_scheduler.steps_per_epoch",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data.valid_stride", "model.metric.init_args.stride"
        )
        parser.link_arguments("optimizer.lr", "lr_scheduler.max_lr")
        parser.link_arguments("trainer.max_epochs", "lr_scheduler.epochs")


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
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, "train.log")
    configure_logging(log_file)
    cli.trainer.fit(cli.model, cli.datamodule)


if __name__ == "__main__":
    main()
