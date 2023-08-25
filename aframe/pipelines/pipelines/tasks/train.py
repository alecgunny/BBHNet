import os

import luigi
from pipelines.constants import CONFIG_DIR, IFOS
from pipelines.tasks.apptainer import AframeApptainerTask


class Train(AframeApptainerTask):
    data_dir = luigi.Parameter(default=os.getenv("DATA_DIR", ""))
    run_dir = luigi.Parameter(default=os.getenv("RUN_DIR", ""))
    config = luigi.Parameter(default=str(CONFIG_DIR / "train.yaml"))
    seed = luigi.IntParameter(default=101588)
    overwrite = luigi.BoolParameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.data_dir:
            raise ValueError("Must specify data directory")
        elif not os.path.exists(self.data_dir):
            raise ValueError(f"Data directory {self.data_dir} does not exist")

        if not self.run_dir:
            raise ValueError("Must specify run root directory")
        os.makedirs(self.run_dir, exist_ok=self.overwrite)

        if not os.path.exists(self.config):
            raise ValueError(f"Config file {self.config} does not exist")

    @property
    def image(self):
        return str(self.root / "container-images" / "train.sif")

    @property
    def gpus(self):
        gpus = os.getenv("CUDA_VISIBLE_DEVICES", "")
        return list(map(int, gpus.split(",")))

    @property
    def log_output(self):
        return False

    @property
    def command(self):
        command = f"""
            python /opt/aframe/aframe/train/train/main.py
                --config {self.config}
                --seed_everything={self.seed}
                --data.ifos=[{','.join(IFOS)}]
                --data.data_dir={self.data_dir}
                --trainer.logger.save_dir={self.run_dir}
        """
        if self.gpus:
            devices = ",".join(map(str, range(len(self.gpus))))
            command += f" --trainer.devices=[{devices}]"
            if len(self.gpus) > 1:
                command += " --trainer.strategy=ddp"
        return command
