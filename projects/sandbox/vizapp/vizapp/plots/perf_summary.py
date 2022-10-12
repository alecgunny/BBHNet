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
        self.hist_type_select = Select(
            title="Histogram type", value="Cumulative", options=["Cumulative", "Binned"]
        )
        
        self.x_axis_select.on_change("value", self.switch_x_axis)
        self.hist_type_select.on_change("value", self.switch_hist_type)
        
        self.p = figure(
            title="Efficiency vs. FAR",
            height=int(0.9 * height),
            width=width,
            x_axis_label="False Alarrm rate [yr\u207B\xB9]",
            y_axis_label="Fraction of true positives",
            x_axis_type="log",
            tools="",
        )
        self.p.toolbar.autohide = True

        self.p.multi_line(
            "x",
            "efficiency",
            line_color="color",
            line_width=2.3,
            line_alpha=0.85,
            legend_field="label",
            source=self.source,
        )
        self.p.legend.location = "top_left"

        
        self.layout = column(self.x_axis_select, self.hist_type_select, self.p)
        self.fars = self.snrs = None

    def update(self, foreground):
        self.snrs = foreground.snrs
        self.fars = foreground.fars
        self.switch_x_axis(None, None, self.x_axis_select.value)

    
    def calc_efficiencies(self, hist_type: str, x_axis_type: str):
        xs, efficiencies, colors, labels = [], [], [], []
        
        snrs = self.snrs
        fars = self.fars
        if x_axis_type == "FAR":
            # snr bins for FAR x axis
            bins = [(0, 5), (5, 20), (20, 100), (100, np.inf)]
        else:
            # far threshold for snr x axis
            bins = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000]
        
        
        for color, vals in zip(palette, bins):
            if x_axis_type == "FAR":
                low, high = vals
                mask = (low <= snrs) & (snrs < high)
                x = np.sort(fars[mask])
                x = np.clip(x, 0.1, np.inf)
                efficiency = (np.arange(len(x)) + 1) / len(x)
                label = f"SNR Range {low}-{high}, N={len(x)}"
            else:
                # bin events in SNR to calculate efficiencies
                if hist_type == "Binned":
                    
                    n_bins = 10
                    
                    mask = fars <= vals
                    detected_snrs = snrs[mask]
                    snr_bins = np.logspace(0, np.log10(max(snrs)), n_bins)
                    detected_counts, _ = np.histogram(detected_snrs, bins=snr_bins)
                    counts, bins = np.histogram(snrs, bins=snr_bins)
                    
                    x = (snr_bins[:-1] + snr_bins[1:]) / 2
                    efficiency = detected_counts / counts
                    label = f"FAR <= {vals} / year, N={len(snrs[mask])}"
                
                # cumulative
                else:
                    mask = fars <= vals
                    x = np.sort(snrs[mask])
                    efficiency = (np.arange(len(x)) + 1) / len(fars)
                    label = f"FAR <= {vals} / year, N={len(x)}"

            xs.append(x)
            efficiencies.append(efficiency)
            colors.append(color)
            labels.append(label)
        
        return xs, efficiencies, colors, labels
        
    def switch_hist_type(self, attr, old, new):
        # histogram type doesn't effect FAR x-axis plot
        if self.x_axis_select.value == "FAR":
            return
        else:
            self.p.title.text = f"Efficiency vs. SNR ({self.hist_type_select.value})"
            xs, efficiencies, colors, labels = self.calc_efficiencies(new, self.x_axis_select.value)
                
        self.source.data["x"] = xs
        self.source.data["efficiency"] = efficiencies
        self.source.data["color"] = colors
        self.source.data["label"] = labels
    
        
        
    def switch_x_axis(self, attr, old, new):
        snrs = self.snrs
        fars = self.fars

        self.p.y_range.start = 0
        if new == "FAR":
            self.p.title.text = "Efficiency vs. FAR (Cumulative)"
            self.p.xaxis.axis_label = "False Alarm Rate [yr^-1]"
        else:
            self.p.title.text = f"Efficiency vs. SNR ({self.hist_type_select.value})"
            self.p.xaxis.axis_label = "Event SNR"

        xs, efficiencies, colors, labels = self.calc_efficiencies(self.hist_type_select.value, new)
 
        self.p.x_range.start = 0.5 * min([x[x > 0].min() for x in xs])
        self.p.x_range.end = 1.5 * max([max(x) for x in xs])

        if new == "SNR":
            max_y = 1.5 * max([max(y) for y in efficiencies])
            self.p.y_range.end = max_y
        else:
            self.p.y_range.end = 1.02

        self.source.data["x"] = xs
        self.source.data["efficiency"] = efficiencies
        self.source.data["color"] = colors
        self.source.data["label"] = labels
