import os

import luigi
from pipelines.tasks.apptainer import ApptainerTask


class GenerateWaveforms(ApptainerTask):
    prior = luigi.Parameter()
    num_signals = luigi.IntParameter()
    output_file = luigi.Parameter()
    sample_rate = luigi.FloatParameter()
    waveform_duration = luigi.FloatParameter()
    waveform_approximant = luigi.Parameter()
    reference_frequency = luigi.FloatParameter()
    minimum_frequency = luigi.FloatParameter()
    waveform_approximant = luigi.Parameter()
    verbose = luigi.BoolParameter(default=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def image(self) -> str:
        default = os.path.expanduser("~/aframe/images")
        root = os.environ.get("AFRAME_CONTAINER_ROOT", default)
        return os.path.join(root, "datagen.sif")

    @property
    def command(self) -> str:
        command = f"""
            python /opt/aframe/aframe/datagen/datagen/waveforms.py
                --prior {self.prior}
                --num-signals {self.num_signals}
                --output-file {self.output_file}
                --sample-rate {self.sample_rate}
                --waveform-duration {self.waveform_duration}
                --waveform-approximant {self.waveform_approximant}
                --reference-frequency {self.reference_frequency}
                --minimum-frequency {self.minimum_frequency}
                --waveform-approximant {self.waveform_approximant}
                --verbose
        """
        return command
