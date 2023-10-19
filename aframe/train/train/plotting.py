from tempfile import NamedTemporaryFile
from math import ceil

from bokeh.io import save
from bokeh.layouts import gridplot
from bokeh.plotting import figure
from bokeh.palettes import Bright5 as palette
from bokeh.resources import CDN


def make_figure(**kwargs):
    default_kwargs = dict(
        height=300,
        width=700,
        tools=""
    )
    default_kwargs.update(kwargs)

    p = figure(**default_kwargs)
    if not default_kwargs["tools"]:
        p.toolbar_location = None

    title = default_kwargs.get("title")
    if title and title.startswith("$$"):
        p.title.text_font_style = "normal"
    return p


def hide_axis(p, axis):
    axis = getattr(p, axis + "axis")
    axis.major_tick_line_color = None
    axis.minor_tick_line_color = None

    # can't set this to 0 otherwise log-axis
    # plots freak out and won't render
    axis.major_label_text_font_size = "1pt"
    axis.major_label_text_color = None


def plot_batch(x, channels, sample_rate):
    nrows = int(len(x)**0.5)
    ncols = ceil(len(x) / nrows)
    t = x.size(-1) / sample_rate

    plots = []
    for i, sample in enumerate(x.cpu().numpy()):
        kwargs = dict(height=200, width=200, tools="box_zoom,reset")
        row, col = divmod(i, ncols)
        if not col:
            kwargs["y_axis_label"] = "Strain"
            kwargs["width"] = 225
        if row == (nrows - 1):
            kwargs["x_axis_label"] = "Time [s]"
            kwargs["width"] = 225

        if plots:
            kwargs["x_range"] = plots[0].x_range
            kwargs["y_range"] = plots[0].y_range

        p = make_figure(**kwargs)
        if col:
            hide_axis(p, "y")
        if row < (nrows - 1):
            hide_axis(p, "x")

        for j, channel in enumerate(sample):
            kwargs = {}
            if not i:
                kwargs["legend_label"] = channels[j]

            p.line(
                t,
                channel,
                line_color=palette[j],
                line_width=1.5,
                line_alpha=0.8,
                **kwargs
            )
        plots.append(p)

    layout = gridplot(
        plots,
        toolbar_location="right",
        ncols=ncols
    )
    return layout


def log_batch(logger, x, channels, sample_rate, title):
    import wandb

    t = [i / sample_rate for i in range(x.size(-1))]
    data = []
    for sample in x.cpu().numpy():
        p = make_figure(
            x_axis_label="Time [s]",
            y_axis_label="Strain",
            tools="box_zoom,reset"
        )
        for channel, color, label in zip(sample, palette, channels):
            p.line(
                t,
                channel,
                line_color=color,
                line_width=1.5,
                line_alpha=0.8,
                legend_label=label
            )
        with NamedTemporaryFile("w+b") as f:
            save(p, f.name, resources=CDN, title=title)
            wandb_html = wandb.Html(f.name)
        data.append([wandb_html])
        logger.log_table(key="batch", columns=[title], data=data)
