import numpy as np
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Select
from bokeh.plotting import figure
from vizapp import palette


class PerfSummaryPlot:
    def __init__(self, height: int, width: int) -> None:
        self.source = ColumnDataSource(
            dict(x=[], efficiency=[], color=[], label=[])
        )

        self.x_axis_select = Select(
            title="X-Axis", value="FAR", options=["FAR", "SNR"]
        )
        self.x_axis_select.on_change("value", self.switch_x_axis)
        self.p = figure(
            title="Efficiency vs. False Alarm Rate",
            height=0.9 * height,
            width=width,
            x_axis_label="False Alarrm rate [yr^-1]",
            y_axis_label="Fraction of true positives",
        )
        self.p.toolbar_autohide = True

        self.p.line(
            "x",
            "efficiency",
            line_color="color",
            line_width=2.3,
            line_alpha=0.85,
            legend_label="label",
            source=self.source,
        )
        self.layout = column(self.x_axis_select, self.p)
        self.fars = self.snrs = None

    def update(self, foreground):
        self.snrs = foreground.snrs
        self.fars = foreground.fars
        self.shift_x_axis(None, None, self.x_axis_select.value)

    def switch_x_axis(self, attr, old, new):
        snrs = self.snrs
        fars = self.fars

        if new == "FAR":
            bins = [(0, 5), (5, 20), (20, 100), (100, np.inf)]
            self.p.x_axis_label = "False Alarm Rate [yr^-1]"
        else:
            bins = [1, 10, 100, 1000, 10000]
            self.p.x_axis_label = "Event SNR"

        min_y, max_y = np.inf, -np.inf
        xs, efficiencies, colors, labels = [], [], [], []
        for color, vals in zip(palette, bins):
            if new == "FAR":
                low, high = vals
                mask = (low <= snrs) & (snrs < high)
                x = np.sort(fars[mask])
                efficiency = (np.arange(len(x)) + 1) / len(x)
            else:
                mask = fars <= vals
                x = np.sort(snrs[mask])
                efficiency = (np.arange(len(x)) + 1) / len(fars)

            if new == "FAR":
                label = f"SNR Range {low}-{high}, N={len(x)}"
            else:
                label = f"FAR <= {vals} / year, N={len(x)}"

            min_y = min(min_y, min(efficiency))
            max_y = max(max_y, max(efficiency))

            xs.append(x)
            efficiencies.append(efficiency)
            colors.append(color)
            labels.append(label)

        self.p.y_range.start = 0.98 * min_y
        self.p.y_range.end = 1.02 * max_y
        self.source.data["x"] = xs
        self.source.data["efficiency"] = efficiencies
        self.source.data["color"] = color
        self.source.data["label"] = labels
