import os
from concurrent.futures import Executor, as_completed
from dataclasses import dataclass, field
from typing import List, Optional, Union

import h5py
import numpy as np
from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters
from bilby.gw.source import lal_binary_black_hole
from bilby.gw.waveform_generator import WaveformGenerator

PATH = Union[str, bytes, os.PathLike]


# define metadata for various types of injection set attributes
# so that they can be easily extended by just annotating your
# new argument with the appropriate type of field
def parameter(default=None):
    default = default or np.array([])
    return field(metadata={"kind": "parameter"}, default=default)


def waveform(default=None):
    default = default or np.array([])
    return field(metadata={"kind": "waveform"}, default=default)


def metadata(default=None):
    return field(metadata={"kind": "metadata"}, default=default)


def _load_with_idx(f: h5py.File, cls: type, idx: Optional[np.ndarray] = None):
    def _try_get(group: str, field: str):
        try:
            group = f[group]
        except KeyError:
            raise ValueError(
                f"Archive {f.filename} has no group {group}"
            ) from None

        try:
            return group[field]
        except KeyError:
            raise ValueError(
                "{} group of archive {} has no dataset {}".format(
                    group, f.filename, field
                )
            ) from None

    kwargs = {}
    for key, attr in cls.__dataclass_fields__.items():
        try:
            kind = attr.metadata["kind"]
        except KeyError:
            raise TypeError(
                f"Couldn't load field {key} with no 'kind' metadata"
            )

        if kind == "metadata":
            value = f.attrs[key][()]
        elif kind not in ("parameter", "waveform"):
            raise TypeError(
                "Couldn't load unknown annotation {} "
                "for field {}".format(kind, key)
            )
        else:
            value = _try_get(kind + "s", key)
            if idx is not None:
                value = value[idx]
            else:
                value = value[:]

        kwargs[key] = value
    return cls(**kwargs)


@dataclass
class InjectionSet:
    def __post_init__(self):
        # get our length up front and make sure that
        # everything that isn't metadata has the same length
        _length = None
        for key, attr in self.__dataclass_fields__.items():
            if attr.metadata["kind"] == "metadata":
                continue
            value = getattr(self, key)

            if _length is None:
                _length = len(value)
            elif len(value) != _length:
                raise ValueError(
                    "Field {} has {} entries, expected {}".format(
                        key, len(value), _length
                    )
                )
        self._length = _length

    def __len__(self):
        return self._length

    def __iter__(self):
        fields = self.__dataclass_fields__
        return map(
            lambda i: {k: self.__dict__[k][i] for k in fields},
            range(len(self)),
        )

    # for slicing and masking sets of parameters/waveforms
    def __getitem__(self, *args, **kwargs):
        init_kwargs = {}
        for key, attr in self.__dataclass_fields__.items():
            value = getattr(self, key)
            if attr.metadata["kind"] != "metadata":
                value = value.__getitem__(*args, **kwargs)
            try:
                len(value)
            except TypeError:
                value = np.array([value])
            init_kwargs[key] = value
        return type(self)(**init_kwargs)

    def _get_group(self, f: h5py.File, name: str):
        return f.get(name) or f.create_group(name)

    def write(self, fname: PATH) -> None:
        with h5py.File(fname, "w") as f:
            f.attrs["length"] = len(self)
            for key, attr in self.__dataclass_fields__.items():
                value = getattr(self, key)
                try:
                    kind = attr.metadata["kind"]
                except KeyError:
                    raise TypeError(
                        f"Couldn't save field {key} with no annotation"
                    )

                if kind == "parameter":
                    params = self._get_group(f, "parameters")
                    params[key] = value
                elif kind == "waveform":
                    waveforms = self._get_group(f, "waveforms")
                    waveforms[key] = value
                elif kind == "metadata":
                    f.attrs[key] = value
                else:
                    raise TypeError(
                        "Couldn't save unknown annotation {} "
                        "for field {}".format(kind, key)
                    )

    @classmethod
    def read(cls, fname: PATH):
        with h5py.File(fname, "r") as f:
            return _load_with_idx(f, cls, None)

    @classmethod
    def sample_from_file(cls, fname, N: int, replace: bool = False):
        """Helper method for when we want to do out-of-memory dataloading
        Can imagine extending to take an arbitrary `weights`
        callable that takes the file object as an input and
        returns sampling weights for each one of the samples,
        so you could e.g. condition sampling on mass
        """
        with h5py.File(fname, "r") as f:
            n = f.attrs["length"]
            if N > n and not replace:
                raise ValueError(
                    "Not enough waveforms to sample without replacement"
                )

            # technically faster in the replace=True case to
            # just do a randint but they're both O(10^-5)s
            # so gonna go for the simpler implementation
            idx = np.random.choice(n, size=(N,), replace=replace)
            return _load_with_idx(f, cls, idx)

    def compare_metadata(self, key, ours, theirs):
        if ours != theirs:
            raise ValueError(
                "Can't append {} with {} value {} "
                "when ours is {}".format(
                    self.__class__.__name__, key, theirs, ours
                )
            )
        return ours

    def append(self, other) -> None:
        if not isinstance(other, type(self)):
            raise TypeError(
                "unsupported operand type(s) for |: '{}' and '{}'".format(
                    type(self), type(other)
                )
            )

        for key, attr in self.__dataclass_fields__.items():
            ours = getattr(self, key)
            theirs = getattr(other, key)
            if attr.metadata["kind"] == "metadata":
                new = self.compare_metadata(key, ours, theirs)
                self.__dict__[key] = new
            else:
                self.__dict__[key] = np.concatenate([ours, theirs])
        self.__post_init__()


@dataclass
class IntrinsicParameterSet(InjectionSet):
    """
    Easy to initialize with:
    params = prior.sample(N)
    params = IntrinsicParameterSet(**params)
    """

    mass_1: np.ndarray = parameter()
    mass_2: np.ndarray = parameter()
    # ... reset of instrinsic params


@dataclass
class InjectionMetadata(InjectionSet):
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
class InjectionParameterSet(InjectionSet):
    """
    Assume GPS times always correspond to un-shifted data
    """

    gps_time: np.ndarray = parameter()
    shift: np.ndarray = parameter()  # 2D with shift values along 1th axis


@dataclass
class SkyLocationParameterSet(InjectionSet):
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
            return _load_with_idx(f, cls, idx)

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


@dataclass
class TimeSlideEventSet(InjectionSet):
    Tb: float = metadata(0)
    decision_statistic: np.ndarray = parameter()
    time: np.ndarray = parameter()

    def compare_metadata(self, key, ours, theirs):
        if key == "Tb":
            return ours + theirs
        return super().compare_metadata(key, ours, theirs)


@dataclass
class EventSet(TimeSlideEventSet):
    shift: np.ndarray = parameter()

    @classmethod
    def from_timeslide(cls, event_set: TimeSlideEventSet, shift: List[float]):
        shifts = np.ndarray([shift] * len(event_set))
        d = {k: getattr(event_set, k) for k in event_set.__dataclass_fields__}
        return cls(shift=shifts, **d)


# inherit from TimeSlideEventSet since injection
# will already have shift information recorded
@dataclass
class RecoveredInjectionSet(TimeSlideEventSet, InterferometerResponseSet):
    @classmethod
    def recover(
        cls,
        events: TimeSlideEventSet,
        responses: InterferometerResponseSet,
        offset: float,
    ):
        # TODO: need an implementation that will
        # also do masking on shifts
        diffs = np.abs(events.time - responses.gps_time[:, None] - offset)
        idx = diffs.argmin(axis=-1)
        events = events[idx]
        kwargs = events.__dict__ | responses.__dict__
        for attr in responses.waveform_fields:
            kwargs.pop(attr)
        return cls(**kwargs)
