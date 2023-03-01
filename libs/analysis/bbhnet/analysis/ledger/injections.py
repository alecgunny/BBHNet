from concurrent.futures import Executor, as_completed
from dataclasses import dataclass, field
from typing import Optional

import h5py
import numpy as np
from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters
from bilby.gw.source import lal_binary_black_hole
from bilby.gw.waveform_generator import WaveformGenerator

from bbhnet.analysis.ledger.ledger import (
    PATH,
    Ledger,
    metadata,
    parameter,
    waveform,
)


@dataclass
class IntrinsicParameterSet(Ledger):
    """
    Easy to initialize with:
    params = prior.sample(N)
    params = IntrinsicParameterSet(**params)
    """

    mass_1: np.ndarray = parameter()
    mass_2: np.ndarray = parameter()
    # ... reset of instrinsic params


@dataclass
class InjectionMetadata(Ledger):
    sample_rate: np.ndarray = metadata()
    duration: np.ndarray = metadata()
    num_injections: int = metadata(default=0)

    def __post_init__(self):
        # verify that all waveforms have the appropriate duration
        super().__post_init__()
        for key, attr in self.__dataclass_fields__.items():
            kind = attr.metadata["kind"]
            if kind != "waveform":
                continue

            duration = field.shape[-1] / self.sample_rate
            if duration != self.duration:
                raise ValueError(
                    "Specified waveform duration of {} but "
                    "waveform '{}' has duration {}".format(
                        self.duration, key, duration
                    )
                )

    @property
    def waveform_fields(self):
        fields = self.__dataclass_fields__.items()
        fields = filter(fields, lambda x: x[1].metadata["kind"] == "waveform")
        return [i[0] for i in fields]

    def compare_metadata(self, key, ours, theirs):
        if key == "num_injections":
            return ours + theirs
        return super().compare_metadata(key, ours, theirs)


@dataclass(frozen=True)
class _WaveformGenerator:
    """Thin wrapper so that we can potentially parallelize this"""

    gen: WaveformGenerator
    sample_rate: float
    waveform_duration: float

    def center(self, waveform):
        dt = self.waveform_duration / 2
        return np.roll(waveform, int(dt * self.sample_rate))

    def __call__(self, params):
        polarizations = self.gen.time_domain_strain(params)

        # could think about stacking then unstacking to
        # make this more efficient
        for key in polarizations.keys():
            polarizations[key] = self.center(polarizations[key])
        return polarizations


@dataclass
class IntrinsicWaveformSet(InjectionMetadata, IntrinsicParameterSet):
    cross: np.ndarray = waveform
    plus: np.ndarray = waveform

    @property
    def waveform_duration(self):
        return self.cross.shape[-1] / self.sample_rate

    def get_waveforms(self) -> np.ndarray:
        return np.stack([self.cross, self.plus])

    @classmethod
    def from_parameters(
        cls,
        params: IntrinsicParameterSet,
        minimum_frequency: float,
        reference_frequency: float,
        sample_rate: float,
        waveform_duration: float,
        waveform_approximant: str,
        ex: Optional[Executor] = None,
    ):
        gen = WaveformGenerator(
            duration=waveform_duration,
            sampling_frequency=sample_rate,
            frequency_domain_source_model=lal_binary_black_hole,
            parameter_conversion=convert_to_lal_binary_black_hole_parameters,
            waveform_arguments={
                "waveform_approximant": waveform_approximant,
                "reference_frequency": reference_frequency,
                "minimum_frequency": minimum_frequency,
            },
        )
        waveform_generator = _WaveformGenerator(
            gen, waveform_duration, sample_rate
        )

        waveform_length = int(sample_rate * waveform_duration)
        polarizations = {
            "plus": np.zeros((len(params), waveform_length)),
            "cross": np.zeros((len(params), waveform_length)),
        }

        # give flexibility if we want to parallelize or not
        if ex is None:
            for i, polars in enumerate(map(waveform_generator, params)):
                for key, value in polars.items():
                    polarizations[key][i] = value
        else:
            futures = ex.map(waveform_generator, params)
            idx_map = {f: i for f, i in zip(futures, len(futures))}
            for f in as_completed(futures):
                i = idx_map.pop(f)
                polars = f.result()
                for key, value in polars.items():
                    polarizations[key][i] = value

        d = {k: getattr(params, k) for k in params.__dataclass_fields__}
        polarizations.update(d)
        polarizations["sample_rate"] = sample_rate
        polarizations["duration"] = waveform_duration
        return cls(**polarizations)


@dataclass
class InjectionParameterSet(Ledger):
    """
    Assume GPS times always correspond to un-shifted data
    """

    gps_time: np.ndarray = parameter()
    shift: np.ndarray = parameter()  # 2D with shift values along 1th axis


@dataclass
class SkyLocationParameterSet(Ledger):
    ra: np.ndarray = parameter()
    dec: np.ndarray = parameter()
    phi: np.ndarray = parameter()
    redshift: np.ndarray = parameter()


@dataclass
class ExtrinsicParameterSet(InjectionParameterSet, SkyLocationParameterSet):
    pass


# note, dataclass inheritance goes from last to first,
# so the ordering of kwargs here would be:
# mass1, mass2, ..., ra, dec, psi, gps_time, shift, sample_rate, h1, l1
@dataclass
class InterferometerResponseSet(
    InjectionMetadata, ExtrinsicParameterSet, IntrinsicParameterSet
):
    def __post_init__(self):
        super().__post_init__()
        self._waveforms = None

    @property
    def waveforms(self) -> np.ndarray:
        if self._waveforms is None:
            fields = sorted(self.waveform_fields)
            waveforms = [getattr(self, i) for i in fields]
            waveforms = np.stack(waveforms, axis=1)
            self._waveforms = waveforms
        return self._waveforms

    @classmethod
    def read(
        cls,
        fname: PATH,
        start: Optional[float] = None,
        end: Optional[float] = None,
    ):
        """
        Similar wildcard behavior in loading. Additional
        kwargs to be able to load data for just a particular
        segment since the slice method below will make copies
        and you could start to run out of memory fast.
        """
        with h5py.File(fname, "r") as f:
            if start is None and end is None:
                idx = None
            else:
                duration = f.attrs["duration"]
                times = f["parameters"]["gps_time"][:]
                mask = True
                if start is not None:
                    mask &= (times + duration / 2) >= start
                if end is not None:
                    mask &= (times - duration / 2) <= end
                idx = np.where(mask)[0]
            return cls._load_with_idx(f, idx)

    def inject(self, x: np.ndarray, start: float):
        """
        Inject waveforms into background array with
        initial timestamp `start`
        """
        stop = start + x.shape[-1] / self.sample_rate
        mask = self.gps_time >= (start - self.duration / 2)
        mask &= self.gps_time <= (stop + self.duration / 2)

        times = self.gps_time[mask]
        waveforms = self.get_waveforms()[mask]

        # potentially pad x to inject waveforms
        # that fall over the boundaries of chunks
        pad = []
        early = (times - self.duration / 2) < start
        if early.any():
            pad.append(early.sum())
        else:
            pad.append(0)

        late = (times + self.duration / 2) > stop
        if late.any():
            pad.append(late.sum())
        else:
            pad.append(0)

        if any(pad):
            x = np.pad(x, pad, axis=1)
        times = times - times[0]

        # create matrix of indices of waveform_size for each waveform
        waveforms = waveforms.transpose((1, 0, 2))
        _, num_waveforms, waveform_size = waveforms.shape
        idx = np.arange(waveform_size) - int(waveform_size // 2)
        idx = idx[None]
        idx = np.repeat(idx, num_waveforms, axis=0)

        # offset the indices of each waveform
        # according to their time offset
        idx_diffs = (times * self.sample_rate).astype("int64")
        idx += idx_diffs[:, None]

        # flatten these indices and the signals out
        # to 1D and then add them in-place all at once
        idx = idx.reshape(-1)
        waveforms = waveforms.reshape(2, -1)
        x[idx] += waveforms
        if any(pad):
            start, stop = pad
            stop = -stop or None
            x = x[start:stop]
        return x


@dataclass
class LigoResponseSet(InterferometerResponseSet):
    h1: np.ndarray = waveform()
    l1: np.ndarray = waveform()