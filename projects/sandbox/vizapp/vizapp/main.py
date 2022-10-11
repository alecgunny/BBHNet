from pathlib import Path

from bokeh.plotting import curdoc
from vizapp.app import VizApp

BBHNET_DIR = Path("/home/alec.gunny/bbhnet")
if __name__ == "__main__":
    app = VizApp(
        timeslides_dir=(
            BBHNET_DIR
            / "results"
            / "faster-training-cleaned-up"
            / "timeslide_injections"
        ),
        data_dir=BBHNET_DIR / "data",
        sample_rate=4096,
        fduration=1,
        valid_frac=0.25,
    )
    curdoc().add_root(app.layout)
