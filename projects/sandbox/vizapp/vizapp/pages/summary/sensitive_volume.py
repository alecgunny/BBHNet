import numpy as np
from bokeh.models import Band, ColumnDataSource, Legend, LegendItem
from bokeh.palettes import Dark2_8 as palette
from bokeh.plotting import figure
from scipy.integrate import quad
from tqdm import trange

SECONDS_PER_MONTH = 3600 * 24 * 30


def get_log_normal_params(mean, std):
    sigma = np.log((std / mean) ** 2 + 1) ** 0.5
    mu = 2 * np.log(mean / (mean**2 + std**2) ** 0.25)
    return mu, sigma


def log_normal_nll(mu, sigma, x):
    exponent = (np.log(x) - mu) ** 2 / (2 * sigma**2)
    norm = x * sigma * (2 * np.pi) ** 0.5
    return exponent + np.log(norm)


class LogNormalProb:
    def __init__(self, mass_1, mass_2, std):
        self.mu1, self.sigma1 = get_log_normal_params(mass_1, std)
        self.mu2, self.sigma2 = get_log_normal_params(mass_2, std)

    def __call__(self, mass_1, mass_2):
        prob_1 = log_normal_nll(self.mu1, self.sigma1, mass_1)
        prob_2 = log_normal_nll(self.mu2, self.sigma2, mass_2)
        return np.exp(-prob_1 - prob_2)


def convert_to_distance(volume):
    dist = 3 * volume / 4 / np.pi
    dist[dist > 0] = dist[dist > 0] ** (1 / 3)
    return dist


class SensitiveVolumePlot:
    def __init__(self, page):
        self.page = page

        max_far_per_month = 10
        Tb = page.app.background.Tb / SECONDS_PER_MONTH
        self.max_events = int(max_far_per_month * Tb)
        self.x = np.arange(1, self.max_events + 1) / Tb
        self.volume = self.get_astrophysical_volume()

        # compute the likelihood of all injections from
        # the run under the prior that generated them
        source = self.page.app.source_prior
        self.num_injections = 0
        num_accepted = len(page.app.foreground)
        self.num_injections += num_accepted

        self.source_probs = np.zeros((num_accepted,))
        mass_1 = page.app.foreground.mass_1
        mass_2 = page.app.foreground.mass_2
        for i in trange(num_accepted):
            sample = {"mass_1": mass_1[i], "mass_2": mass_2[i]}
            self.source_probs[i] = source.prob(sample)

        # this includes rejected injections
        num_rejected = len(page.app.rejected_params)
        self.num_injections += num_rejected

        self.source_rejected_probs = np.zeros((num_rejected,))
        mass_1 = page.app.rejected_params.mass_1
        mass_2 = page.app.rejected_params.mass_2
        for i in trange(num_rejected):
            sample = {"mass_1": mass_1[i], "mass_2": mass_2[i]}
            self.source_rejected_probs[i] = source.prob(sample)

    def volume_element(self, z):
        cosmology = self.page.app.cosmology
        return cosmology.differential_comoving_volume(z).value / (1 + z)

    def get_astrophysical_volume(self):
        z_prior = self.page.app.source_prior["redshift"]
        zmin, zmax = z_prior.minimum, z_prior.maximum
        volume, _ = quad(self.volume_element, zmin, zmax)

        try:
            dec_prior = self.page.app.source_prior["dec"]
        except KeyError:
            theta_min, theta_max = 0, np.pi
        else:
            theta_max = np.pi / 2 - dec_prior.minimum
            theta_min = np.pi / 2 - dec_prior.maximum
        omega = -2 * np.pi * (np.cos(theta_max) - np.cos(theta_min))
        return volume * omega

    def initialize_sources(self):
        mass_combos = [
            (35, 35),
            (35, 20),
            (20, 20),
            (20, 10),
            # (10, 10)
        ]
        self.color_map = {i: j for i, j in zip(mass_combos, palette)}
        self.color_map["MDC"] = "#000000"

        foreground = self.page.app.foreground
        rejected = self.page.app.rejected_params

        self.probs = {}
        for combo in mass_combos:
            calc = LogNormalProb(*combo, std=1)
            prob = calc(foreground.mass_1, foreground.mass_2)
            rejected_prob = calc(rejected.mass_1, rejected.mass_2)
            self.probs[combo] = (prob, rejected_prob)

        self.line_source = ColumnDataSource(dict(x=self.x))
        self.band_source = ColumnDataSource(dict(x=self.x))
        self.update()

    def get_sd_data(self, mu, std):
        # convert them both to volume units
        volume = mu * self.volume
        std = std * self.volume

        # convert volume to distance, and use
        # the distance of the upper and lower
        # volume values as our distance bands
        distance = convert_to_distance(volume)
        low = convert_to_distance(volume - std)
        high = convert_to_distance(volume + std)
        return distance, low, high

    def make_label(self, key):
        if key == "MDC":
            return "MDC (no IS)"
        return "Log Normal {}/{}".format(*key)

    def update(self):
        line_data = {}
        band_data = {}

        # compute all of the thresholds we'll use for
        # estimating sensitive volume up front, removing
        # any background events that are rejected due
        # to any active vetoes
        background = self.page.app.background.detection_statistic
        background = background[~self.page.app.veto_mask]
        thresholds = np.sort(background)[-self.max_events :][::-1]

        # mask will have shape
        # (self.max_events, num_foreground_events)
        foreground = self.page.app.foreground.detection_statistic
        mask = foreground >= thresholds[:, None]
        mask = mask.astype("int")

        # calculate the SD under the injected prior
        mu = mask.sum(axis=1) / self.num_injections
        var = mu * (1 - mu) / self.num_injections
        std = var**0.5

        distance, low, high = self.get_sd_data(mu, std)
        label = self.make_label("MDC")

        line_data[label] = distance
        band_data[label + " low"] = low
        band_data[label + " high"] = high

        for combo, (probs, rejected_probs) in self.probs.items():
            weights = probs / self.source_probs
            rejected_weights = rejected_probs / self.source_rejected_probs

            # normalize all the weights up front
            # to make the downstream calculations simple
            norm = weights.sum() + rejected_weights.sum()
            weights = weights / norm
            rejected_weights = rejected_weights / norm

            # calculate the weighted average
            # probability of detection
            recovered_weights = weights * mask
            mu = recovered_weights.sum(axis=1)

            # calculate variance of this estimate
            var_summand = weights * (mask - mu[:, None])
            std = (var_summand**2).sum(axis=1) ** 0.5

            # convert them both to volume units
            distance, low, high = self.get_sd_data(mu, std)

            # add all of these to our sources
            label = self.make_label(combo)
            line_data[label] = distance
            band_data[label + " low"] = low
            band_data[label + " high"] = high

        self.line_source.data.update(line_data)
        self.band_source.data.update(band_data)

    def plot_data(self, p, key):
        label = self.make_label(key)
        color = self.color_map[key]

        r = p.line(
            x="x",
            y=label,
            line_width=2,
            line_color=color,
            source=self.line_source,
        )

        band = Band(
            base="x",
            lower=label + " low",
            upper=label + " high",
            fill_color=color,
            line_color=color,
            fill_alpha=0.3,
            line_width=0.8,
            source=self.band_source,
        )
        p.add_layout(band)
        return LegendItem(renderers=[r], label=label)

    def get_layout(self, height, width):
        pad = 0.01 * (self.x.max() - self.x.min())
        p = figure(
            height=height,
            width=width,
            title=(
                r"$$\text{Importance Sampled Sensitive Distance "
                r"with 1-}\sigma\text{ deviation}$$"
            ),
            x_axis_label=r"$$\text{False Alarm Rate [months}^{-1}\text{]}$$",
            y_axis_label=r"$$\text{Sensitive Distance [Mpc]}$$",
            x_range=(self.x.min() - pad, self.x.max() + pad),
            tools="save",
        )

        item = self.plot_data(p, "MDC")
        items = [item]
        for combo in self.probs:
            item = self.plot_data(p, combo)
            items.append(item)

        legend = Legend(items=items)
        p.add_layout(legend, "right")
        return p
