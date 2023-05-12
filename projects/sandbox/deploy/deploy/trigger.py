import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Literal, Optional, Union

import numpy as np
from ligo.gracedb.rest import GraceDb

from bbhnet.analysis.ledger.events import TimeSlideEventSet

Gdb = Literal["playground", "test", "production"]
SECONDS_PER_YEAR = 31556952


@dataclass
class Event:
    time: float
    detection_statistic: float
    far: float


class Searcher:
    def __init__(
        self,
        outdir: Path,
        far_per_day: float,
        inference_sampling_rate: float,
        refractory_period: float,
        offset: float,
    ) -> None:
        logging.debug("Loading background measurements")
        background_file = outdir / "infer" / "background.h5"
        self.background = TimeSlideEventSet.read(background_file)

        num_events = int(far_per_day * self.background.Tb / 3600 / 24)
        if not num_events:
            raise ValueError(
                "Background livetime {}s not enough to detect "
                "events with false alarm rate of {} per day".format(
                    self.background.Tb, far_per_day
                )
            )

        events = np.sort(self.background.detection_statistic)
        self.threshold = events[-num_events]
        logging.info(
            "FAR {}/day threshold is {:0.3f}".format(
                far_per_day, self.threshold
            )
        )

        self.inference_sampling_rate = inference_sampling_rate
        self.refractory_period = refractory_period
        self.offset = offset
        self.last_detection_time = time.time()
        self.detecting = False

    def check_refractory(self, value):
        time_since_last = time.time() - self.last_detection_time
        if time_since_last < self.refractory_period:
            logging.warning(
                "Detected event with detection statistic {:0.3f} "
                "but it has only been {}s since last detection, "
                "so skipping".format(value, time_since_last)
            )
            return True
        return False

    def build_event(self, value: float, t0: float, idx: int):
        if self.check_refractory(value):
            return None

        timestamp = t0 + self.inference_sampling_rate * idx
        timestamp += self.offset

        far = self.background.far(value)
        far /= SECONDS_PER_YEAR

        logging.info(
            "Event coalescence time found to be {:0.3f} "
            "with FAR {:0.3e}Hz".format(timestamp, far)
        )
        self.last_detection_time = time.time()
        return Event(timestamp, value, far)

    def search(self, y: np.ndarray, t0: float) -> Optional[Event]:
        max_val = y.max()
        logging.debug(f"Max integrated value from frame {t0} is {max_val}")

        if not self.detecting and max_val <= self.threshold:
            return None
        elif not self.detecting:
            logging.info(
                "Detected event with detection statistic>={max_val:0.3f}"
            )

            idx = np.argmax(y)
            if idx < (len(y) - 1):
                return self.build_event(max_val, t0, idx)
            else:
                self.detecting = True
                return None
        elif self.detecting:
            # TODO: the assumption here is that if the
            # peak wasn't in the last frame, it has to
            # be in this frame. This is true with the
            # current BBHNet, but will obviously have to
            # be updated with longer kernels
            idx = np.argmax(y)
            self.detecting = False
            return self.build_event(max_val, t0, idx)


@dataclass
class LocalGdb:
    write_dir: Path

    def createEvent(self, filecontents: dict, **_):
        filename = "event-{timestamp}.json".format(**filecontents)
        filename = self.write_dir / filename
        logging.info(f"Submitting trigger to file {filename}")
        with open(filename, "w") as f:
            json.dump(filecontents, f)
        return filename


class Trigger:
    def __init__(self, server: Union[Gdb, Path]) -> None:
        if isinstance(server, Path):
            self.gdb = LocalGdb(server)
            return

        if server in ["playground", "test"]:
            server = f"https://gracedb-{server}.ligo.org/api/"
        elif server == "production":
            server = "https://gracedb.ligo.org/api/"
        else:
            raise ValueError(f"Unknown server {server}")
        self.gdb = GraceDb(service_url=server)

    def submit(self, event: Event, ifos: List[str]):
        filename = "event.json"
        event = asdict(event)
        event["IFOs"] = ifos
        filecontents = str(event)

        # alternatively we can write a file to disk,
        # pass that path to the filename argument,
        # and set filecontents=None
        response = self.gdb.createEvent(
            group="CBC",
            pipeline="BBHNet",
            filename=filename,
            search="BBH",
            filecontents=filecontents,
        )
        return response
