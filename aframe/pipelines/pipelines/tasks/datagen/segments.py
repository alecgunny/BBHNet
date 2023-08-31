import os

import luigi
from pipelines.configs import aframe
from pipelines.tasks.apptainer import AframeApptainerTask


class GenerateSegments(AframeApptainerTask):
    start = luigi.FloatParameter()
    stop = luigi.FloatParameter()
    state_flag = luigi.Parameter()
    minimum_length = luigi.FloatParameter()
    maximum_length = luigi.FloatParameter()
    ifos = luigi.ListParameter(default=aframe().ifos)
    output_file = luigi.Parameter(default=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def image(self) -> str:
        default = os.path.expanduser("~/aframe/images")
        root = os.environ.get("AFRAME_CONTAINER_ROOT", default)
        return os.path.join(root, "datagen.sif")

    @property
    def environment(self) -> dict:
        return {"X509_USER_PROXY": os.getenv("X509_USER_PROXY", "")}

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
                --maximum-length {self.maximum_length}
                --ifos {' '.join(self.ifos)}
                --output-file {self.output_file}
        """
        return command

    def output(self):
        return luigi.LocalTarget(self.output_file)
