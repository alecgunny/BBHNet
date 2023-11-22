import os
import re

import luigi
from pipelines.configs import Defaults, aframe, wandb
from pipelines.tasks.apptainer import AframeApptainerTask, K8ServiceTask


class Train(AframeApptainerTask):
    data_dir = luigi.Parameter(default=os.getenv("DATA_DIR", ""))
    run_dir = luigi.Parameter(default=os.getenv("RUN_DIR", ""))
    config = luigi.Parameter(default="")
    seed = luigi.IntParameter(default=101588)
    devices = luigi.Parameter(default=os.getenv("CUDA_VISIBLE_DEVICES", ""))
    overwrite = luigi.BoolParameter()
    wandb = luigi.BoolParameter()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not self.data_dir:
            raise ValueError("Must specify data directory")
        elif not os.path.exists(self.data_dir):
            raise ValueError(f"Data directory {self.data_dir} does not exist")

        if not self.run_dir:
            raise ValueError("Must specify run root directory")
        os.makedirs(self.run_dir, exist_ok=self.overwrite)

        self.config = self.config or Defaults.TRAIN
        if not os.path.exists(self.config):
            raise ValueError(f"Config file {self.config} does not exist")

    @property
    def image(self) -> str:
        default = os.path.expanduser("~/aframe/images")
        root = os.environ.get("AFRAME_CONTAINER_ROOT", default)
        return os.path.join(root, "train.sif")

    @property
    def gpus(self) -> list[int]:
        return list(map(int, self.devices.split(",")))

    @property
    def log_output(self) -> bool:
        return False

    @property
    def environment(self) -> dict[str, str]:
        env = {}
        if wandb().api_key:
            env["WANDB_API_KEY"] = wandb().api_key
        return env

    def configure_wandb(self, command: str) -> str:
        command += " --trainer.logger=WandbLogger"
        command += " --trainer.logger.job_type=train"

        config = wandb()
        for key in ["name", "entity", "project", "group", "tags"]:
            value = getattr(config, key)
            if value:
                command += f" --trainer.logger.{key}={value}"
        return command

    @property
    def command(self) -> str:
        command = f"""
            python /opt/aframe/aframe/train/train/main.py
                --config {self.config}
                --seed_everything={self.seed}
                --data.ifos=[{','.join(aframe().ifos)}]
                --data.data_dir={self.data_dir}
        """

        if self.gpus:
            devices = ",".join(map(str, range(len(self.gpus))))
            command += f" --trainer.devices=[{devices}]"
            if len(self.gpus) > 1:
                command += " --trainer.strategy=ddp"

        if self.wandb and not wandb().api_key:
            raise ValueError(
                "Can't run W&B experiment without specifying an API key."
            )
        elif self.wandb:
            command = self.configure_wandb(command)
        command += f" --trainer.logger.save_dir={self.run_dir}"
        return command


class Tune(K8ServiceTask, Train):
    gpus_per_job = luigi.IntParameter(default=1)
    cpus_per_job = luigi.IntParameter(default=8)
    num_samples = luigi.IntParameter(default=10)

    @property
    def gpus(self):
        return []

    def command(self):
        command = super().command()
        command.replace("main.py", "tune.py")

        command += f" --tune.address={self.ip}:10001"
        for param in ["gpus_per_job", "cpus_per_job", "num_samples"]:
            value = getattr(self, param)
            command += f" --tune.{param}={value}"

        command = re.sub("--trainer.logger.name=\S+", "", command)
        name = wandb().name
        if name:
            command += f" --tune.name {name}"
        return command
