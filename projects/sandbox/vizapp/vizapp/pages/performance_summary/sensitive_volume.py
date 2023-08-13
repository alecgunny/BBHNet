import logging
from collections import defaultdict

import numpy as np
from bokeh.models import ColumnDataSource, HoverTool, Legend, LegendItem
from bokeh.palettes import Dark2_8 as palette
from bokeh.plotting import figure
from scipy.integrate import quad
from tqdm import tqdm

from aframe.priors.priors import log_normal_masses

SECONDS_PER_MONTH = 3600 * 24 * 30


def make_tooltip(field):
    return f"@{{{field}}}{{0.}}"


def get_prob(prior, ledger):
    sample = dict(mass_1=ledger.mass_1, mass_2=ledger.mass_2)
    return prior.prob(sample, axis=0)


def convert_to_distance(volume):
    if volume == 0:
        return 0
    dist = 3 * volume / 4 / np.pi
    return dist**(1 / 3)
    # dist[dist > 0] = dist[dist > 0] ** (1 / 3)
    # return dist


class SensitiveVolumePlot:
    def __init__(self, page):
        self.page = page
        self.logger = logging.getLogger("Sensitive Volume")

        max_far_per_month = 1000
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
        self.logger.info("Computing likelihood of injections under source")
        self.source_probs = get_prob(source, page.app.foreground)

        # this includes rejected injections
        self.logger.info("Computing likelihood of rejected under source")
        num_rejected = len(page.app.rejected_params)
        self.num_injections += num_rejected
        self.source_rejected_probs = get_prob(source, page.app.rejected_params)

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
        self.color_map["End O3"] = "#000000"

        foreground = self.page.app.foreground
        rejected = self.page.app.rejected_params

        self.weights = {}
        cosmology = self.page.app.cosmology
        for combo in mass_combos:
            self.logger.info(f"Computing likelihoods under {combo} log normal")
            prior, _ = log_normal_masses(*combo, sigma=1, cosmology=cosmology)
            prob = get_prob(prior, foreground)
            rejected_prob = get_prob(prior, rejected)

            weight = prob / self.source_probs
            rejected_weights = rejected_prob / self.source_rejected_probs
            norm = weight.sum() + rejected_weights.sum()
            self.weights[combo] = weight / norm

        self.line_source = ColumnDataSource(dict(x=self.x))

        band_x = np.concatenate([self.x, self.x[::-1]])
        self.band_source = ColumnDataSource(dict(x=band_x))
        self.update()

    def get_err(self, V, dV):
        denom = np.sqrt(4 * np.pi) * 3 * V
        return dV / denom**(2/3)

    def get_sd_data(self, mu, std):
        # convert them both to volume units
        volume = mu * self.volume
        std = std * self.volume

        # convert volume to distance, and use
        # the distance of the upper and lower
        # volume values as our distance bands
        err = self.get_err(volume, std)
        distance = convert_to_distance(volume)
        low = distance - err
        high = distance + err
        return distance, low, high

    def make_label(self, key):
        if key == "End O3":
            return "End O3 (no IS)"
        return "Log Normal {}/{}".format(*key)

    def compute_sd_for_threshold(self, threshold):
        foreground = self.page.app.foreground.detection_statistic
        mask = foreground >= threshold

        # calculate the SD under the injected prior
        mu = mask.sum() / self.num_injections
        var = mu * (1 - mu) / self.num_injections
        std = var**0.5

        distance, low, high = self.get_sd_data(mu, std)
        label = self.make_label("End O3")

        line_data = {label: distance, label + " err": (high - low) / 2}
        band_data = {label: (low, high)}
        for combo, weight in self.weights.items():
            # calculate the weighted average
            # probability of detection
            mu = weight[mask].sum()

            # calculate variance of this estimate
            var_summand = weight * (mask - mu)
            std = (var_summand**2).sum() ** 0.5

            # convert them both to volume units
            distance, low, high = self.get_sd_data(mu, std)

            # add all of these to our sources
            label = self.make_label(combo)
            line_data[label] = distance
            line_data[label + " err"] = (high - low) / 2
            band_data[label] = (low, high)
        return line_data, band_data

    def update(self):
        line_data = defaultdict(list)
        low_data = defaultdict(list)
        high_data = defaultdict(list)

        # compute all of the thresholds we'll use for
        # estimating sensitive volume up front, removing
        # any background events that are rejected due
        # to any active vetoes
        background = self.page.app.background.detection_statistic
        background = background[~self.page.app.veto_mask]
        background = background[~np.isnan(background)]
        thresholds = np.sort(background)[-self.max_events :][::-1]

        for threshold in tqdm(thresholds):
            ld, bd = self.compute_sd_for_threshold(threshold)
            for key, value in ld.items():
                line_data[key].append(value)

            for key, (low, high) in bd.items():
                low_data[key].append(low)
                high_data[key].insert(0, high)

        band_data = {}
        for key, value in low_data.items():
            band_data[key] = np.concatenate([value, high_data[key]])
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

        band = p.patch(
            x="x",
            y=label,
            fill_color=color,
            fill_alpha=0.3,
            line_width=0.8,
            line_color=color,
            line_alpha=0.3,
            source=self.band_source,
        )

        tooltip = "{} +/- {} Mpc".format(
            make_tooltip(label), make_tooltip(label + " err")
        )
        hover = HoverTool(
            renderers=[r],
            tooltips=[("FAR", "@x"), (label, tooltip)],
            point_policy="snap_to_data",
            line_policy="nearest",
        )
        return hover, LegendItem(renderers=[r, band], label=label)

    def get_layout(self, height, width):
        pad = (self.x.max() / self.x.min()) ** 0.01
        p = figure(
            height=height,
            width=width,
            title=(
                r"$$\text{Importance Sampled Sensitive Distance "
                r"with 1-}\sigma\text{ deviation}$$"
            ),
            x_axis_label=r"$$\text{False Alarm Rate [months}^{-1}\text{]}$$",
            y_axis_label=r"$$\text{Sensitive Distance [Mpc]}$$",
            x_range=(self.x.min() / pad, self.x.max() * pad),
            x_axis_type="log",
            tools="save",
        )

        hover, item = self.plot_data(p, "End O3")
        items = [item]
        p.add_tools(hover)
        for combo in self.weights:
            hover, item = self.plot_data(p, combo)
            items.append(item)
            p.add_tools(hover)

        legend = Legend(items=items, click_policy="mute")
        p.add_layout(legend, "right")
        return p
