import os

from lightning.pytorch.cli import LightningCLI
from train.model import Aframe

from aframe.logging import configure_logging


def main():
    cli = LightningCLI(
        model_class=Aframe,
        seed_everything_default=101588,
        run=False,
        parser_kwargs={"default_env": True},
        save_config_kwargs={"overwrite": True},
    )

    save_dir = cli.trainer.logger.save_dir
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, "train.log")
    configure_logging(log_file)
    cli.trainer.fit(cli.model)


if __name__ == "__main__":
    main()
