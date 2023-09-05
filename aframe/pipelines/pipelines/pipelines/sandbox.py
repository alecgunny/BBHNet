import os

import luigi
from pipelines.tasks.datagen import (
    GenerateBackground,
    GenerateTimeslideWaveforms,
    GenerateWaveforms,
)


# base config that tasks can reference
class Base(luigi.Config):
    data_dir = luigi.Parameter(os.getenv("DATA_DIR", ""))
    train_start = luigi.FloatParameter(default=1240579783)
    train_stop = luigi.FloatParameter(default=1241443783)
    test_stop = luigi.FloatParameter(default=1242114183)
    state_flag = luigi.Parameter(default="DCS-ANALYSIS_READY_C01:1")
    minimum_length = luigi.FloatParameter(default=1024)
    maximum_length = luigi.FloatParameter(default=20000)
    channel = luigi.Parameter(default="DCS-CALIB_STRAIN_CLEAN_C01")
    ifos = luigi.ListParameter(default=["H1", "L1"])
    sample_rate = luigi.FloatParameter(default=2048)
    training_prior = luigi.Parameter(
        default="aframe.priors.priors.end_o3_ratesandpops"
    )
    highpass = luigi.FloatParameter(default=32)
    dev = luigi.BoolParameter(default=True)


class Waveforms(luigi.Config):
    prior = luigi.Parameter(default=Base().training_prior)
    num_signals = luigi.IntParameter()
    sample_rate = luigi.FloatParameter()
    waveform_duration = luigi.FloatParameter()
    waveform_approximant = luigi.Parameter()
    reference_frequency = luigi.FloatParameter()
    minimum_frequency = luigi.FloatParameter()
    waveform_approximant = luigi.Parameter()


class TimeSlides(luigi.Config):
    shifts = luigi.ListParameter(default=[0, 1])
    spacing = luigi.FloatParameter(default=60)
    buffer = luigi.FloatParameter(default=10)
    snr_threshold = luigi.FloatParameter(default=4)


# here is where we'll define the pipeline;
# can run via
# `poetry run
# luigi --module pipelines.pipelines.sandbox Sandbox --local-scheduler`
class Sandbox(luigi.WrapperTask):
    def requires(self):
        yield GenerateBackground(
            root=os.path.join(Base().data_dir, "train"),
            submit_dir=os.path.join(
                Base().data_dir, "train", "condor", "background"
            ),
            sample_rate=Base().sample_rate,
            channel=Base().channel,
            start=Base().train_start,
            stop=Base().train_stop,
            state_flag=Base().state_flag,
            minimum_length=Base().minimum_length,
            maximum_length=Base().maximum_length,
            ifos=Base().ifos,
            dev=True,
        )

        yield GenerateWaveforms(
            prior=Base().training_prior(),
            num_signals=Waveforms().num_signals,
            output_file=os.path.join(Base().data_dir, "train", "signals.h5"),
            sample_rate=Waveforms().sample_rate,
            waveform_duration=Waveforms().waveform_duration,
            waveform_approximant=Waveforms().waveform_approximant,
            reference_frequency=Waveforms().reference_frequency,
            minimum_frequency=Waveforms().minimum_frequency,
        )

        # will launch dependency tree
        # query test segments ->
        # generate train background -> generate timeslide waveforms
        yield GenerateTimeslideWaveforms(
            start=Base().train_stop,
            stop=Base().test_stop,
            minimum_length=Base().minimum_length,
            maximum_length=Base().maximum_length,
            ifos=Base().ifos,
            shifts=TimeSlides().shifts,
            spacing=TimeSlides.spacing(),
            buffer=TimeSlides.buffer(),
            prior=Base.training_prior(),
            minimum_frequency=Waveforms().minimum_frequency,
            reference_frequency=Waveforms().reference_frequency,
            sample_rate=Base().sample_rate,
            waveform_duration=Waveforms().waveform_duration,
            waveform_approximant=Waveforms().waveform_approximant,
            highpass=Base().highpass,
            snr_threshold=TimeSlides().snr_threshold,
            root=Base().data_dir,
            dev=Base().dev,
        )
