from pathlib import Path

import luigi
from luigi.target import FileSystemTarget


class AframeDataDirectory(FileSystemTarget):
    fs = luigi.local_target.LocalFileSystem()

    def __init__(self, path):
        self.root = Path(path)
        self.background = self.root / "background"
        self.background_condor = self.root / "condor" / "background"

        self.timeslide = self.root / "timeslide_waveforms"
        self.timeslide_condor = self.root / "condor" / "timeslide_waveforms"

    def makedirs(self):
        # make train and test directories for storing data
        self.background.mkdir(parents=True, exist_ok=True)
        self.timeslide.mkdir(parents=True, exist_ok=True)

        # make condor directories for storing condor submission files
        self.background_condor.mkdir(parents=True, exist_ok=True)
        self.timeslide_condor.mkdir(parents=True, exist_ok=True)

    def exists(self):
        exists = self.background.exists() and self.timeslide.exists()
        exists &= (
            self.background_condor.exists() and self.timeslide_condor.exists()
        )
        return exists

    def open(self, mode="r"):
        raise NotImplementedError

    def __str__(self):
        return self.path

    def __repr__(self):
        return self.path


class BuildDirectory(luigi.Task):
    root = luigi.Parameter()

    def run(self):
        self.output().makedirs()

    def output(self):
        return AframeDataDirectory(self.root)
