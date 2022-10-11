from typing import Optional

import numpy as np
from bokeh.layouts import column
from bokeh.models import (
    BoxSelect,
    ColumnDataSource,
    HoverTool,
    LogAxis,
    Range1d,
)
from bokeh.plotting import figure
from vizapp import palette


def find_glitches(events, times):
    unique_times, counts = np.unique(times, return_counts=True)
    mask = counts > 1
    unique_times, counts = unique_times[mask], counts[mask]

    centers = []
    for t in unique_times:
        values = events[times == t]
        centers.append(np.median(values))
    return unique_times, counts, centers


class BackgroundPlot:
    def __init__(
        self,
        height: int,
        width: int,
        event_inspector,
        norm: Optional[float] = None,
    ) -> None:
        self.configure_sources()
        self.configure_plots(height, width)
        self.event_inspector = event_inspector
        self.norm = norm

    def configure_plots(self, height: int, width: int):
        self.distribution_plot = figure(
            height=height // 2,
            width=width,
            y_axis_type="log",
            x_axis_label="Detection statistic",
            y_axis_label="Survival function",
            tools="",
        )
        self.distribution_plot.toolbar_autohide = True
        self.distribution_plot.yaxis.axis_label_text_color = palette[0]

        self.distribution_plot.vbar(
            "center",
            top="top",
            width="width",
            fill_color=palette[0],
            line_color="#000000",  # palette[0],
            fill_alpha=0.4,
            line_alpha=0.6,
            line_width=0.5,
            selected_fill_alpha=0.6,
            selected_line_alpha=0.8,
            unselected_fill_alpha=0.2,
            unselected_line_alpha=0.3,
            legend_label="Background",
            source=self.bar_source,
        )

        # TODO: BoxSelect args and callback
        box_select = BoxSelect(dimension="width")
        box_select.on_change("indices", self.update_time_plot)

        self.distribution_plot.extra_y_ranges = {"SNR": Range1d(1, 10)}
        axis = LogAxis(
            axis_label="SNR",
            axis_label_text_color=palette[1],
            y_range_name="SNR",
        )
        self.distribution_plot.add_layout(axis, "right")

        r = self.distribution_plot.circle(
            "detection_statistic",
            "snr",
            size="size",
            fill_color=palette[1],
            line_color=palette[1],
            line_width=0.5,
            fill_alpha=0.2,
            line_alpha=0.4,
            y_range_name="SNR",
            legend_label="Events",
            source=self.foreground_source,
        )

        hover = HoverTool(
            tooltips=[
                ("Hanford GPS time", "@{event_time}{0.000}"),
                ("Shift", "@shift"),
                ("SNR", "@snr"),
                ("Detection statistic", "@{detection_statistic}"),
                ("Chirp Mass", "@{chirp_mass}"),
            ],
            renderers=[r],
        )
        self.distribution_plot.add_tools(hover)

        self.background_plot = figure(
            height=height // 2,
            width=width,
            title="",
            x_axis_label="GPS Time [s]",
            y_axis_label="Detection statistic",
            tools="",
        )
        self.distribution_plot.toolbar_autohide = True

        self.background_plot.circle(
            "x",
            "detection_statistic",
            fill_color="color",
            fill_alpha=0.5,
            line_color="color",
            line_alpha=0.7,
            hover_fill_color="color",
            hover_fill_alpha=0.7,
            hover_line_color="color",
            hover_line_alpha=0.9,
            size="size",
            source=self.background_source,
        )

        hover = HoverTool(
            tooltips=[
                ("GPS time", "@{event_time}{0.000}"),
                ("Detection statistic", "@{detection_statistic}"),
                ("Count", "@count"),
            ]
        )
        self.background_plot.add_tools(hover)

        # TODO: hover callbacks

        self.layout = column([self.distribution_plot, self.background_plot])

    def configure_sources(self):
        self.bar_source = ColumnDataSource(dict(center=[], top=[], width=[]))

        self.foreground_source = ColumnDataSource(
            dict(
                detection_statistic=[],
                event_time=[],
                shift=[],
                snr=[],
                chirp_mass=[],
                size=[],
            )
        )

        self.background_source = ColumnDataSource(
            dict(
                x=[],
                event_time=[],
                detection_statistic=[],
                color=[],
                label=[],
                count=[],
                shift=[],
                size=[],
            )
        )

    def update_source(self, source, **kwargs):
        for key, value in kwargs.items():
            source.data[key] = value

    def update(self, foreground, background, norm):
        title = (
            "Distribution of {} background events from "
            "{:0.2f} days worth of data; SNR vs. detection "
            "statistic of {} injections overlayed"
        ).format(
            len(background.events),
            background.Tb / 3600 / 24,
            len(foreground.event_times),
        )
        self.distribution_plot.title = title

        self.norm = norm
        hist, bins = np.histogram(background, bins=100)
        hist = np.cumsum(hist[::-1])[::-1]

        self.update_source(
            self.bar_source,
            center=(bins[:-1] + bins[1:]) / 2,
            top=hist,
            width=0.95 * (bins[1:] - bins[:-1]),
        )

        self.update_source(
            self.foreground_source,
            detection_statistic=foreground.detection_statistics,
            event_time=foreground.event_times,
            shift=foreground.shifts,
            snr=foreground.snrs,
            chirp_mass=foreground.chirps,
            size=foreground.chirps / 10,
        )

        # clear the background plot until we select another
        # range of detection characteristics to plot
        self.update_source(
            self.background_source,
            x=[],
            event_time=[],
            detection_statistic=[],
            color=[],
            label=[],
            count=[],
            size=[],
        )
        self.background_plot.title = (
            "Select detection characteristic range above"
        )
        self.background_plot.xaxis.axis_label = "GPS Time [s]"

    def update_background(self, attr, old, new):
        stats = self.background_source.data["detection_statistic"]
        threshold = min(stats[new])
        mask = stats >= threshold

        events = stats[mask]
        h1_times = self.background_source.data["event_time"][mask]
        shifts = self.background_source.data["shift"][mask][:, 1]
        l1_times = h1_times + shifts

        unique_h1_times, h1_counts, h1_centers = find_glitches(
            events, h1_times
        )
        unique_l1_times, l1_counts, l1_centers = find_glitches(
            events, l1_times
        )

        centers = h1_centers + l1_centers
        times = np.concatenate([unique_h1_times, unique_l1_times])
        counts = np.concatenate([h1_counts, l1_counts])
        colors = [palette[0]] * len(h1_counts) + [palette[1]] * len(l1_counts)
        labels = ["Hanford"] * len(h1_counts) + ["Livingston"] * len(l1_counts)

        t0 = h1_times.min()
        self.update_source(
            self.background_source,
            x=times - t0,
            event_time=times,
            detection_statistic=centers,
            color=colors,
            label=labels,
            count=counts,
            size=counts / 1.5,
        )

    def update_event_inspector(self, idx):
        event_time = self.background_source.data["event_time"][idx]
        shift = self.background_source.data["shift"][idx]
        self.event_inspector.update(event_time, "background", shift, self.norm)
