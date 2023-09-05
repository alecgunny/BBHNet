import os

import luigi
import numpy as np
from luigi.util import inherits
from pipelines.tasks.apptainer import ApptainerTask, CondorApptainerTask
from pipelines.tasks.datagen import GenerateBackground, GenerateSegments


@inherits(GenerateSegments)
class CondorizeTimeslideWaveforms(ApptainerTask):
    root = luigi.Parameter()
    Tb = luigi.FloatParameter()
    max_shift = luigi.FloatParameter()
    psd_length = luigi.FloatParameter()

    def output(self):
        return luigi.LocalTarget(os.path.join(self.root, "parameters.txt"))

    def requires(self):
        return self.clone(GenerateSegments)

    def command(self):
        command = f"""
            python /opt/aframe/aframe/datagen/datagen/condorize.py
                    --segment-path {self.input()}
                    --Tb {self.Tb}
                    --max-shift {self.max_shift}
                    --psd-length {self.psd_length}
        """
        return command


@inherits(GenerateBackground, CondorizeTimeslideWaveforms)
class GenerateTimeslideWaveforms(CondorApptainerTask):
    start = luigi.FloatParameter()
    stop = luigi.FloatParameter()
    minimum_length = luigi.FloatParameter()
    maximum_length = luigi.FloatParameter()
    ifos = luigi.ListParameter()
    shifts = luigi.ListParameter()
    spacing = luigi.FloatParameter()
    buffer = luigi.FloatParameter()
    prior = luigi.Parameter()
    minimum_frequency = luigi.FloatParameter()
    reference_frequency = luigi.FloatParameter()
    sample_rate = luigi.FloatParameter()
    waveform_duration = luigi.FloatParameter()
    waveform_approximant = luigi.Parameter()
    highpass = luigi.FloatParameter()
    snr_threshold = luigi.FloatParameter()
    root = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.background_dir = os.path.join(self.root, "background")
        self.output_dir = os.path.join(self.root, "timeslide_waveforms")
        self.submit_dir = os.path.join()
        self._segments = None

    def output(self):
        return [
            luigi.LocalTarget(os.path.join(self.output_dir, name))
            for name in ["waveforms.h5", "rejected-parameters.h5"]
        ]

    def requires(self):
        # requires testing segments and background data for snr thresholding
        return [
            self.clone(GenerateBackground),
            self.clone(CondorizeTimeslideWaveforms),
        ]

    @property
    def name(self):
        return "timeslide_waveforms"

    @property
    def queue(self):
        return f"queue start,stop,shift from {self.submit_dir}/parameters.txt"

    @property
    def image(self) -> str:
        default = os.path.expanduser("~/aframe/images")
        root = os.environ.get("AFRAME_CONTAINER_ROOT", default)
        return os.path.join(root, "datagen.sif")

    @property
    def command(self):
        command = f"""
            python /opt/aframe/aframe/datagen/datagen/timeslide_waveforms.py
                    --start $(start)
                    --stop $(stop)
                    --ifos {" ".join(self.ifos)}
                    --shifts $(shifts)
                    --psd-file $(psd_file)
                    --spacing {self.spacing}
                    --buffer {self.buffer}
                    --prior {self.prior}
                    --minimum-frequency {self.minimum_frequency}
                    --reference-frequency {self.reference_frequency},
                    --sample-rate {self.sample_rate}
                    --waveform-duration {self.waveform_duration}
                    --waveform-approximant {self.waveform_approximant}
                    --highpass {self.highpass}
                    --snr-threshold {self.snr_threshold}
                    --output-file {self.output_file}
                    --log-file {self.log_file}
                    --verbose
        """
        return command

    @property
    def segments(self):
        if self._segments is None:
            self._segments = np.loadtxt(self.input()[1])
        return self._segments

    def run(self):
        super().run()
