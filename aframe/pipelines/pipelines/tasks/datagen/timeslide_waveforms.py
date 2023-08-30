import os

import luigi
from pipelines.tasks.apptainer import CondorApptainerTask


class GenerateTimeslideWaveforms(CondorApptainerTask):
    start = luigi.FloatParameter()
    stop = luigi.FloatParameter()
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
    output_file = luigi.Parameter()
    log_file = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
                    --start {self.start}
                    --stop {self.stop}
                    --ifos {" ".join(self.ifos)}
                    --shifts {" ".join(self.shifts)}
                    --background-dir {self.background_dir}
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
                    --output_file {self.output_file}
                    --log-file {self.log_file}
                    --verbose
        """
        return command
