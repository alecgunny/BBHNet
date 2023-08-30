import os

import luigi
from pipelines.configs import aframe
from pipelines.tasks.apptainer import AframeApptainerTask


class Segments(AframeApptainerTask):
    data_dir = luigi.Parameter(default=os.getenv("DATA_DIR", ""))
    start = luigi.FloatParameter()
    stop = luigi.FloatParameter()
    state_flag = luigi.Parameter()
    minimum_length = luigi.FloatParameter()
    ifos = luigi.ListParameter(default=aframe().ifos)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.data_dir:
            raise ValueError("Must specify data directory")
        elif not os.path.exists(self.data_dir):
            raise ValueError(f"Data directory {self.data_dir} does not exist")

    @property
    def image(self) -> str:
        default = os.path.expanduser("~/aframe/images")
        root = os.environ.get("AFRAME_CONTAINER_ROOT", default)
        return os.path.join(root, "datagen.sif")

    @property
    def log_output(self) -> bool:
        return False

    @property
    def command(self) -> str:
        command = f"""
            python /opt/aframe/aframe/datagen/datagen/segments.py
                --start {self.start}
                --stop {self.stop}
                --state-flag {self.state_flag}
                --minimum-length {self.minimum_length}
                --ifos {' '.join(self.ifos)}
                --data-dir {self.data_dir}
        """
        return command

    @property
    def output(self):
        return luigi.LocalTarget(self.data_dir / "segments.txt")
