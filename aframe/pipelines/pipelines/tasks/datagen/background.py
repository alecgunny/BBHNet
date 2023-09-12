import os

import luigi
import numpy as np
from luigi.util import inherits
from pipelines.tasks.apptainer import CondorApptainerTask
from pipelines.tasks.datagen.segments import GenerateSegments

from aframe.datagen import make_fname


@inherits(GenerateSegments)
class GenerateCondorParameters(luigi.Task):
    root = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.background_dir = os.path.join(self.root, "background")
        self.parameter_file = os.path.join(
            self.root, "condor", "background", "parameters.txt"
        )
        self._segments = None

    @property
    def segments(self):
        if self._segments is None:
            self._segments = np.loadtxt(self.input().open("r"))
        return self._segments

    @property
    def paths(self):
        starts = self.segments[:, 0]
        stops = self.segments[:, 1]
        durations = stops - starts
        paths = [
            make_fname("background", start, duration)
            for start, duration in zip(starts, durations)
        ]
        return paths

    def output(self):
        return luigi.LocalTarget(self.parameter_file)

    def requires(self):
        return self.clone(GenerateSegments)

    @property
    def image(self) -> str:
        default = os.path.expanduser("~/aframe/images")
        root = os.environ.get("AFRAME_CONTAINER_ROOT", default)
        return os.path.join(root, "datagen.sif")

    @property
    def command(self):
        command = f"""
            python /opt/aframe/aframe/datagen/datagen/background/condorize.py
                --segment-path {str(self.input().path)}
                --background-dir {str(self.background_dir)}
                --output-file {str(self.output()[-1].path)}
        """

        return command


# SubmitGenerateBackground now "inherits" all parameters from GenerateSegments
# see https://luigi.readthedocs.io/en/stable/api/luigi.util.html
@inherits(GenerateSegments)
class SubmitGenerateBackground(CondorApptainerTask):
    root = luigi.Parameter()
    sample_rate = luigi.FloatParameter()
    channel = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def name(self):
        return "generate_background"

    @property
    def job_kwargs(self):
        return {
            "name": "$(ProcId)",
            "submit_name": self.name,
            "error": os.path.join(self.submit_dir),
            "output": os.path.join(self.submit_dir),
            "log": os.path.join(self.submit_dir),
        }

    @property
    def queue(self):
        return "start,stop,writepath from parameters.txt"

    @property
    def command(self):
        command = f"""
            python /opt/aframe/aframe/datagen/datagen/background/background.py
                --start $(start)
                --stop $(stop)
                --channel {self.channel}
                --ifos {' '.join(self.ifos)}
                --sample-rate {self.sample_rate}
                --write-path $(writepath)
        """
        return command

    @property
    def image(self) -> str:
        default = os.path.expanduser("~/aframe/images")
        root = os.environ.get("AFRAME_CONTAINER_ROOT", default)
        return os.path.join(root, "datagen.sif")


# This is apparently a common patter in Luigi for
# dynamic requirements (
# e.g. the paths of the output background files depend on the queried segments)
# Have a "wrapper" task that requires the dynamic depdency.
# Then in the run method, parse the dependency and pass to downstream task
@inherits(SubmitGenerateBackground)
class GenerateBackground(luigi.Task):
    root = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.segment_file = os.path.join(self.root, "segments.txt")

    @property
    def segments(self):
        return np.loadtxt(self.input().open("r"))

    def requires(self):
        # clone will pass common parameters to GenerateSegments.
        # need to explicitly pass constructed output_file
        # since this is not a common parameter
        return self.clone(GenerateSegments)

    def write_parameters(self, parameters: np.ndarray):
        with open(os.path.join(self.submit_dir, "parameters.txt"), "w") as f:
            f.write("start,stop,writepath\n")
            for start, stop, writepath in parameters:
                f.write(f"{start},{stop},{writepath}\n")

    def validate_segments(
        self,
    ):
        # determine which segments need to be generated
        generate = []
        for i, (start, stop) in enumerate(self.segments):
            duration = stop - start
            fname = make_fname("background", start, duration)
            write_path = os.path.join(self.root, "background", fname)

            if not os.path.exists(write_path):
                generate.append(
                    [
                        start,
                        stop,
                        write_path,
                    ]
                )

        return generate

    def run(self):
        # validate segments
        parameters = self.validate_segments()
        # write parameters.txt
        self.write_parameters(parameters)
        yield self.clone(SubmitGenerateBackground)
