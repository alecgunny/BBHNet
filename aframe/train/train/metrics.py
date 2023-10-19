from typing import Optional

import torch


class ShiftedPearsonCorrelation(torch.nn.Module):
    def __init__(self, max_shift: int):
        super().__init__()
        self.max_shift = max_shift

    def forward(self, x, y):
        batch, channels, dim = x.shape

        # window x before padding so that we
        # don't introduce any frequency artifacts
        # window = torch.hann_window(dim, device=x.device)
        # x = x * window

        # pad x along time dimension so that it has shape
        # batch x channels x (dim + 2 * max_shift)
        pad = (self.max_shift, self.max_shift)
        x = torch.nn.functional.pad(x, pad)

        # the following is just some magic to unroll
        # x into dim-length windows along its time axis
        # batch x channels x 1 x (dim + 2 * max_shift)
        x = x.unsqueeze(2)

        # batch x (channels * num_windows) x 1 x dim
        num_windows = 2 * self.max_shift + 1
        x = torch.nn.functional.unfold(x, (1, num_windows))

        # batch x channels x num_windows x dim
        x = x.reshape(batch, channels, num_windows, dim)

        # num_windows x batch x channels x dim
        x = x.transpose(0, 2).transpose(1, 2)

        # now compute the correlation between
        # each one of these windows of x and
        # the single window of y
        # de-mean
        x = x - x.mean(-1, keepdims=True)
        y = y - y.mean(-1, keepdims=True)

        # num_windows x batch x channels
        corr = (x * y).sum(axis=-1)
        norm = (x**2).sum(-1) * (y**2).sum(-1)

        return corr / norm**0.5


class ChiSq(torch.nn.Module):
    def __init__(
        self,
        num_bins: int,
        fftlength: float,
        sample_rate: float,
        highpass: Optional[float] = None
    ) -> None:
        super().__init__()

        self.sample_rate = sample_rate
        self.num_bins = num_bins
        bins = torch.arange(num_bins + 1) / num_bins
        self.register_buffer("bins", bins)

        self.df = 1 / fftlength
        self.num_freqs = int(fftlength * sample_rate // 2 + 1)
        freqs = torch.arange(self.num_freqs) / fftlength
        self.register_buffer("freqs", freqs)

        if highpass is not None:
            mask = freqs >= highpass
            self.register_buffer("mask", mask)
        else:
            self.mask = None

    def get_snr_integral(self, htilde, psd=None, stilde=None):
        stilde = htilde if stilde is None else stilde
        integrand = htilde * stilde.conj()
        if psd is not None:
            integrand /= psd
        snr = (4 * integrand * self.df).abs().cumsum(-1)
        return snr

    def make_indices(self, batch_size, num_channels):
        """
        Helper function for selecting arbitrary indices
        along the last axis of our batches by building
        tensors of repeated index selectors for the
        batch and channel axes.
        """
        idx0 = torch.arange(batch_size)
        idx0 = idx0.view(-1, 1, 1).repeat(1, num_channels, self.num_bins)

        idx1 = torch.arange(num_channels)
        idx1 = idx1.view(1, -1, 1).repeat(batch_size, 1, self.num_bins)
        return idx0, idx1

    def get_snr_per_bin(self, qtilde, stilde, edges, psd=None):
        """
        For a normalized frequency template qtilde and
        frequency-domain strain measurement stilde, measure
        the SNR in the bins between the specified edges
        (whose last dimension should be one greater than the
        number of bins).
        """

        snr_integral = self.get_snr_integral(qtilde, psd, stilde)
        total_snr = snr_integral[:, :, -1:]

        batch_size, num_channels, _ = snr_integral.shape
        idx0, idx1 = self.make_indices(batch_size, num_channels)

        right = snr_integral[idx0, idx1, edges[:, :, 1:]]
        left = snr_integral[idx0, idx1, edges[:, :, :-1]]
        return total_snr, right - left

    def partition_frequencies(self, htilde, psd=None):
        """
        Compute the edges of the frequency bins that would
        (roughly) evenly break up the optimal SNR of the
        template. Normalize the template by its maximum
        SNR to keep "just be loud everywhere" from being
        a strategy.
        """
        snr_integral = self.get_snr_integral(htilde, psd)
        total_snr = snr_integral[:, :, -1:]
        bins = self.bins * total_snr
        edges = torch.searchsorted(snr_integral, bins, side="right")
        edges = edges.clamp(0, snr_integral.size(-1) - 1)

        qtilde = htilde / total_snr**0.5
        return qtilde, edges

    def interpolate_psd(self, psd):
        # have to scale the interpolated psd to ensure
        # that the integral of the power remains constant
        factor = (psd.size(-1) / self.num_freqs)**2
        psd = torch.nn.functional.interpolate(
            psd, size=self.num_freqs, mode="linear"
        )
        return psd * factor

    def forward(self, htilde, stilde, psd: Optional[torch.Tensor] = None):
        if psd is not None and psd.size(-1) != self.num_freqs:
            psd = self.interpolate_psd(psd)
            if self.mask is not None:
                psd = psd[:, :, self.mask]

        if self.mask is not None:
            htilde = htilde[:, :, self.mask]
            stilde = stilde[:, :, self.mask]

        qtilde, edges = self.partition_frequencies(htilde, psd)
        snr, snr_per_bin = self.get_snr_per_bin(qtilde, stilde, edges, psd)

        chisq_summand = snr_per_bin - snr / self.num_bins
        chisq_summand = chisq_summand**2
        chisq = self.num_bins * chisq_summand.sum(-1) / (self.num_bins - 1)
        return snr[:, :, 0], chisq


class MatchedFilterLoss(torch.nn.Module):
    def __init__(
        self,
        max_shift: float,
        sample_rate: float,
        chisq_num_bins: Optional[int] = None,
        chisq_highpass: Optional[float] = None,
        chisq_kernel_length: Optional[float] = None,
        chisq_penalty: float = 1,
        lowpass_cutoff: Optional[float] = None,
        lowpass_penalty: float = 100,
    ) -> None:
        super().__init__()
        max_shift = int(max_shift * sample_rate)
        self.correlation = ShiftedPearsonCorrelation(max_shift)
        self.sample_rate = sample_rate

        if chisq_num_bins is not None:
            if chisq_kernel_length is None:
                raise ValueError(
                    "Must specify a kernel length for chi squared "
                    "loss term if a number of bins is specified"
                )
            self.chisq = ChiSq(
                chisq_num_bins,
                chisq_kernel_length,
                sample_rate=sample_rate,
                highpass=chisq_highpass
            )
            self.chisq_penalty = chisq_penalty
        else:
            self.chisq_penalty = self.chisq = None

        self.lowpass_cutoff = lowpass_cutoff
        self.lowpass_penalty = lowpass_penalty

    def fft(self, x):
        return torch.fft.rfft(x.double(), dim=-1) / self.sample_rate

    def center(self, x_hat):
        xmin = x_hat.min(axis=-1, keepdims=True).values
        xmax = x_hat.max(axis=-1, keepdims=True).values
        return 2 * (x_hat - xmin) / (xmax - xmin) - 1

    def forward(self, x_hat, x):
        # scale each channel's prediction to -1 to 1
        # to account for lack of amplitude info
        x_hat = self.center(x_hat)

        # compute the correlation at multiple shifts,
        # then take the max across all shifts. Scale
        # that inversely from (-1, 1) to (inf, 0)
        corr = self.correlation(x_hat, x)
        corr = corr.max(dim=0).values
        loss = (1 - corr) / (1 + corr)

        if self.chisq is not None or self.lowpass_cutoff is not None:
            # include some loss terms that depend on the
            # frequency content of the generated template
            htilde = self.fft(x_hat)

            # first, apply a chi-squared correction that
            # penalizes the template for not correctly
            # predicting the spread of frequency content
            # in the injected waveform
            if self.chisq is not None:
                stilde = self.fft(x)
                _, chisq = self.chisq(htilde, stilde)
                chisq *= self.chisq_penalty
                loss *= (1 + chisq**3)**(1 / 6)

            # then apply a penalty for having frequency
            # content above some cutoff to avoid noisy
            # templates that accidentally match
            if self.lowpass_cutoff is not None:
                num_freqs = htilde.size(-1)
                freqs = torch.linspace(0, 0.5, num_freqs) * self.sample_rate
                df = self.sample_rate / 2 / (num_freqs - 1)
                mask = freqs >= self.lowpass_cutoff
                lowpass_loss = (htilde[:, :, mask] * df).sum(-1)
                loss += self.lowpass_penalty * lowpass_loss

        # return a loss per channel, per sample
        return loss
