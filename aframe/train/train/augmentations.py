from typing import List, Optional, Union

import torch

from ml4gw import gw
from ml4gw.distributions import Cosine, PowerLaw, Uniform


class ChannelSwapper(torch.nn.Module):
    """
    Data augmentation module that randomly swaps channels
    of a fraction of batch elements.

    Args:
        frac:
            Fraction of batch that will have channels swapped.
    """

    def __init__(self, frac: float = 0.5):
        super().__init__()
        self.frac = frac

    def forward(self, X):
        num = int(X.shape[0] * self.frac)
        indices = []
        if num > 0:
            num = num if not num % 2 else num - 1
            num = max(2, num)
            channel = torch.randint(X.shape[1], size=(num // 2,)).repeat(2)
            # swap channels from the first num / 2 elements with the
            # second num / 2 elements
            indices = torch.arange(num)
            target_indices = torch.roll(indices, shifts=num // 2, dims=0)
            X[indices, channel] = X[target_indices, channel]
        return X, indices


class ChannelMuter(torch.nn.Module):
    """
    Data augmentation module that randomly mutes 1 channel
    of a fraction of batch elements.

    Args:
        frac:
            Fraction of batch that will have channels muted.
    """

    def __init__(self, frac: float = 0.5):
        super().__init__()
        self.frac = frac

    def forward(self, X):
        num = int(X.shape[0] * self.frac)
        indices = []
        if num > 0:
            channel = torch.randint(X.shape[1], size=(num,))
            indices = torch.randint(X.shape[0], size=(num,))
            X[indices, channel] = torch.zeros(X.shape[-1], device=X.device)

        return X, indices


class SignalInverter(torch.nn.Module):
    """
    Data augmentation that randomly inverts data in a kernel

    Args:
        prob:
            Probability that a kernel is inverted
    """

    def __init__(self, prob: float = 0.5):
        super().__init__()
        self.prob = prob

    def forward(self, X):
        if self.training:
            mask = torch.rand(size=X.shape[:-1]) < self.prob
            X[mask] *= -1
        return X


class SignalReverser(torch.nn.Module):
    """
    Data augmentation that randomly reverses data in a kernel

    Args:
        prob:
            Probability that a kernel is reversed
    """

    def __init__(self, prob: float = 0.5):
        super().__init__()
        self.prob = prob

    def forward(self, X):
        if self.training:
            mask = torch.rand(size=X.shape[:-1]) < self.prob
            X[mask] = X[mask].flip(-1)
        return X


class SnrRescaler(torch.nn.Module):
    """
    Module that calculates SNRs of injections relative
    to a given ASD and performs augmentation of the waveform
    dataset by rescaling injections such that they have SNRs
    given by `target_snrs`. If this argument is `None`, each
    injection is randomly matched with and scaled to the SNR
    of a different injection from the batch.
    """

    def __init__(
        self,
        sample_rate: float,
        highpass: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.highpass = highpass

    def forward(
        self,
        responses: gw.WaveformTensor,
        psds: torch.Tensor,
        target_snrs: Union[gw.ScalarTensor, float, None],
    ) -> gw.WaveformTensor:
        # we can either specify one PSD for all batch
        # elements, or a PSD for each batch element
        if psds.ndim > 2 and len(psds) != len(responses):
            raise ValueError(
                "Background PSDs must either be two dimensional "
                "or have a PSD specified for every element in the "
                "batch. Expected {}, found {}".format(
                    len(responses), len(psds)
                )
            )

        # interpolate the number of PSD frequency bins down
        # to the value expected by the shape of the waveforms
        num_freqs = responses.size(-1) // 2 + 1
        if psds.size(-1) != num_freqs:
            if psds.ndim == 2:
                psds = psds[None]
                reshape = True
            else:
                reshape = False

            psds = torch.nn.functional.interpolate(psds, size=(num_freqs,))
            if reshape:
                psds = psds.view(-1, num_freqs)

        # compute the SNRs of the existing signals
        snrs = gw.compute_network_snr(
            responses, psds, self.sample_rate, self.highpass
        )

        if target_snrs is None:
            # if we didn't specify any target SNRs, then shuffle
            # the existing SNRs of the waveforms as they stand
            idx = torch.randperm(len(snrs))
            target_snrs = snrs[idx]
        elif not isinstance(target_snrs, torch.Tensor):
            # otherwise if we provided just a float, assume
            # that it's a lower bound on the desired SNR levels
            target_snrs = snrs.clamp(target_snrs, 1000)

        # reweight the amplitude of the IFO responses
        # in order to achieve the target SNRs
        target_snrs.to(snrs.device)
        weights = target_snrs / snrs
        return responses * weights.view(-1, 1, 1)


class SnrSampler:
    """
    Randomly sample values from a power law distribution,
    initially defined with a minimum of `max_min_snr`, a
    maximum of `max_snr`, and an exponent of `alpha` (see
    `ml4gw.distributions.PowerLaw` for details). The
    distribution will gradually change to have a minimum
    of `min_min_snr` over the course of `decay_steps` steps.

    The ending distribution was chosen as an approximate
    empirical match to the SNR distribution of signals
    generated by `aframe.priors.end_o3_ratesandpops` and
    injected in O3 noise. This curriculum training of
    SNRs is intended to aid the network in learning
    low SNR events.
    """

    def __init__(
        self,
        max_min_snr: float,
        min_min_snr: float,
        max_snr: float,
        alpha: float,
        decay_steps: int,
    ):
        self.max_min_snr = max_min_snr
        self.min_min_snr = min_min_snr
        self.max_snr = max_snr
        self.alpha = alpha
        self.decay_steps = decay_steps
        self._step = 0

        self.dist = PowerLaw(max_min_snr, max_snr, alpha)

    def __call__(self, N):
        return self.dist(N)

    def step(self):
        self._step += 1
        if self._step > self.decay_steps:
            return

        frac = self._step / self.decay_steps
        diff = self.max_min_snr - self.min_min_snr
        new = self.max_min_snr - frac * diff

        self.dist.x_min = new
        self.dist.normalization = new ** (-self.alpha + 1)
        self.dist.normalization -= self.max_snr ** (-self.alpha + 1)


class WaveformProjector(torch.nn.Module):
    def __init__(
        self,
        ifos: List[str],
        sample_rate: float,
        highpass: Optional[float] = None,
    ) -> None:
        super().__init__()
        tensors, vertices = gw.get_ifo_geometry(*ifos)
        self.register_buffer("tensors", tensors)
        self.register_buffer("vertices", vertices)

        self.sample_rate = sample_rate
        self.rescaler = SnrRescaler(sample_rate, highpass)

    def forward(
        self,
        dec: torch.Tensor,
        psi: torch.Tensor,
        phi: torch.Tensor,
        snrs: Union[torch.Tensor, float, None] = None,
        psds: Optional[torch.Tensor] = None,
        **polarizations: torch.Tensor
    ) -> torch.Tensor:
        responses = gw.compute_observed_strain(
            dec,
            psi,
            phi,
            detector_tensors=self.tensors,
            detector_vertices=self.vertices,
            sample_rate=self.sample_rate,
            **polarizations,
        )
        if snrs is not None:
            if psds is None:
                raise ValueError(
                    "Must specify background PSDs if projecting "
                    "to target SNR"
                )
            responses = self.rescaler(responses, psds, snrs)
        return responses


class WaveformSampler(torch.nn.Module):
    """
    TODO: modify this to sample waveforms from disk, taking
    an index sampler object so that DDP training can sample
    different waveforms for each device.
    """

    def __init__(
        self, inject_prob: float, **polarizations: torch.Tensor
    ) -> None:
        super().__init__()
        self.dec = Cosine()
        self.psi = Uniform(0, torch.pi)
        self.phi = Uniform(-torch.pi, torch.pi)

        self.inject_prob = inject_prob
        self.num_waveforms = None
        for polar, x in polarizations.items():
            if self.num_waveforms is None:
                self.num_waveforms = len(x)
            if len(x) != self.num_waveforms:
                raise ValueError(
                    "Expected all waveform polarizations to have "
                    "length {}, but {} polarization has length {}".format(
                        self.num_waveforms, polar, len(x)
                    )
                )
        self.polarizations = polarizations

    def forward(self, X):
        # sample which batch elements of X we're going to inject on
        rvs = torch.rand(size=X.shape[:1], device=X.device)
        mask = rvs < self.inject_prob
        N = mask.sum().item()

        # sample sky parameters for each injections
        dec = self.dec(N).to(X.device)
        psi = self.psi(N).to(X.device)
        phi = self.phi(N).to(X.device)

        # now sample the actual waveforms we want to inject
        idx = torch.randperm(self.num_waveforms)[:N]
        polarizations = {}
        for polarization, waveforms in self.polarizations.items():
            waveforms = waveforms[idx]
            polarizations[polarization] = waveforms.to(dec.device)
        return dec, psi, phi, polarizations, mask
