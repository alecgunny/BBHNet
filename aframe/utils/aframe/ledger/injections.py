from dataclasses import dataclass
from typing import Optional

import h5py
import numpy as np
from aframe.ledger.ledger import PATH, Ledger, metadata, parameter, waveform


@dataclass
class CBC(Ledger):
    """
    Easy to initialize with:
    params = prior.sample(N)
    params = IntrinsicParameterSet(**params)
    """

    mass_1: np.ndarray = parameter()
    mass_2: np.ndarray = parameter()
    redshift: np.ndarray = parameter()
    a_1: np.ndarray = parameter()
    a_2: np.ndarray = parameter()
    tilt_1: np.ndarray = parameter()
    tilt_2: np.ndarray = parameter()
    phi_12: np.ndarray = parameter()
    phi_jl: np.ndarray = parameter()


@dataclass
class SkyLocation(Ledger):
    ra: np.ndarray = parameter()
    dec: np.ndarray = parameter()
    psi: np.ndarray = parameter()
    theta_jn: np.ndarray = parameter()
    phase: np.ndarray = parameter()


@dataclass
class WaveformMetadata(Ledger):
    sample_rate: np.ndarray = metadata()

    def __post_init__(self):
        super().__post_init__()
        if self.sample_rate is None and self._length > 0:
            raise ValueError(
                "Must specify sample rate if not "
                "initializing {} as empty container ".format(
                    self.__class__.__name__
                )
            )

    @property
    def waveform_fields(self):
        fields = self.__dataclass_fields__.items()
        fields = filter(lambda x: x[1].metadata["kind"] == "waveform", fields)
        return sorted([i[0] for i in fields])

    def get_waveforms(self):
        if not len(self):
            return np.array([])

        waveforms = [getattr(self, i) for i in self.waveform_fields]
        return np.stack(waveforms, axis=1)

    @property
    def duration(self):
        if not len(self):
            return None

        field = self.waveform_fields[0]
        return getattr(self, field).shape[-1] / self.sample_rate


@dataclass
class WaveformPolarization(SkyLocation, CBC, WaveformMetadata):
    cross: np.ndarray = waveform
    plus: np.ndarray = waveform


@dataclass
class Event(Ledger):
    gps_time: np.ndarray = parameter()
    shift: np.ndarray = parameter()  # 2D with shift values along 1th axis

    def get_shift(self, shift):
        mask = self.shift == shift
        if self.shift.ndim == 2:
            mask = mask.all(axis=-1)
        return self[mask]

    def get_times(
        self, start: Optional[float] = None, end: Optional[float] = None
    ):
        if start is None and end is None:
            raise ValueError("Must specify one of start or end")

        mask = True
        if start is not None:
            mask &= self.gps_time >= start
        if end is not None:
            mask &= self.gps_time < end
        return self[mask]


@dataclass
class Injection(Event, SkyLocation, CBC):
    snr: np.ndarray = parameter()


@dataclass
class InjectionCampaign(Injection):
    num_injections: int = metadata(default=0)

    def __post_init__(self):
        super().__post_init__()
        if self.num_injections < self._length:
            raise ValueError(
                "{} has fewer total injections {} than "
                "number of waveforms {}".format(
                    self.__class__.__name__, self.num_injections, self._length
                )
            )

    @classmethod
    def compare_metadata(cls, key, ours, theirs):
        if key == "num_injections":
            if ours is None:
                return theirs
            elif theirs is None:
                return ours
            return ours + theirs
        return Ledger.compare_metadata(key, ours, theirs)


@dataclass
class IfoResponse(InjectionCampaign, WaveformMetadata):
    def __post_init__(self):
        super().__post_init__()
        self._waveforms = None

    @classmethod
    def _raise_bad_shift_dim(cls, fname, dim1, dim2):
        raise ValueError(
            "Specified shifts with {} dimensions, but "
            "{} from file {} has {} dimensions".format(
                dim1, cls.__name__, fname, dim2
            )
        )

    @classmethod
    def read(
        cls,
        fname: PATH,
        start: Optional[float] = None,
        end: Optional[float] = None,
        shifts: Optional[float] = None,
    ):
        """
        Similar wildcard behavior in loading. Additional
        kwargs to be able to load data for just a particular
        segment since the slice method below will make copies
        and you could start to run out of memory fast.
        """
        with h5py.File(fname, "r") as f:
            if all([i is None for i in [start, end, shifts]]):
                return cls._load_with_idx(f, None)

            duration = f.attrs["duration"]
            times = f["parameters"]["gps_time"][:]

            mask = True
            if start is not None:
                mask &= (times + duration / 2) >= start
            if end is not None:
                mask &= (times - duration / 2) <= end
            if shifts is not None:
                shifts = np.array(shifts)
                ndim = shifts.ndim

                fshifts = f["parameters"]["shift"][:]
                f_ndim = fshifts.ndim
                if f_ndim == 2:
                    if ndim == 1:
                        shifts = shifts[None]
                    elif ndim != 2:
                        cls._raise_bad_shift_dim(fname, ndim, f_ndim)
                elif f_ndim == 1:
                    if ndim != 1:
                        cls._raise_bad_shift_dim(fname, ndim, f_ndim)
                    fshifts = fshifts[:, None]
                    shifts = shifts[:, None]
                else:
                    cls._raise_bad_shift_dim(fname, ndim, f_ndim)

                if fshifts.shape[-1] != shifts.shape[-1]:
                    raise ValueError(
                        "Specified {} shifts when {} ifos "
                        "are present in {} {}".format(
                            shifts.shape[-1],
                            fshifts.shape[-1],
                            cls.__name__,
                            fname,
                        )
                    )

                shift_mask = False
                for shift in shifts:
                    shift_mask |= (fshifts == shift).all(axis=-1)
                mask &= shift_mask

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

        if not mask.any():
            return x

        # cache waveforms so we're not catting each time
        if self._waveforms is None:
            self._waveforms = self.get_waveforms()

        times = self.gps_time[mask]
        waveforms = self._waveforms[mask]

        # potentially pad x to inject waveforms
        # that fall over the boundaries of chunks
        pad = []
        earliest = (times - self.duration / 2 - start).min()
        if earliest < 0:
            num_early = int(-earliest * self.sample_rate)
            pad.append(num_early)
            start += earliest
        else:
            pad.append(0)

        latest = (times + self.duration / 2 - stop).max()
        if latest > 0:
            num_late = int(latest * self.sample_rate)
            pad.append(num_late)
        else:
            pad.append(0)

        if any(pad):
            x = np.pad(x, [(0, 0)] + [tuple(pad)])
        times = times - start

        # create matrix of indices of waveform_size for each waveform
        waveforms = waveforms.transpose((1, 0, 2))
        _, num_waveforms, waveform_size = waveforms.shape
        idx = np.arange(waveform_size) - waveform_size // 2
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
        x[:, idx] += waveforms
        if any(pad):
            start, stop = pad
            stop = -stop or None
            x = x[:, start:stop]
        return x


@dataclass
class LigoResponse(IfoResponse):
    h1: np.ndarray = waveform()
    l1: np.ndarray = waveform()
