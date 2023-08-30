import logging
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from textwrap import dedent
from typing import Sequence

import luigi
from pycondor import Job

logger = logging.getLogger("luigi-interface")


class ImageNotFound(Exception):
    pass


class ApptainerTask(luigi.Task):
    @property
    def image(self) -> str:
        raise NotImplementedError

    @property
    def command(self) -> str:
        return "echo hello world"

    @property
    def environment(self) -> dict:
        return {}

    @property
    def binds(self) -> dict:
        return {}

    @property
    def gpus(self) -> Sequence[int]:
        return []

    @property
    def log_output(self) -> bool:
        return True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not os.path.exists(self.image):
            raise ImageNotFound(
                f"Couldn't find container image {self.image} locally"
            )
        self._binds = self.binds
        self.__logger = logger

    def run(self):
        env = {}
        for key, value in self.environment.items():
            env[f"APPTAINERENV_{key}"] = value

        cmd = ["apptainer", "exec"]
        for source, dest in self._binds.items():
            cmd.extend(["--bind", f"{source}:{dest}"])

        if self.gpus:
            cmd.append("--nv")

            gpus = ",".join(map(str, self.gpus))
            env["APPTAINERENV_CUDA_VISIBLE_DEVICES"] = gpus

        cmd.append(self.image)

        command = dedent(self.command).replace("\n", " ")
        command = shlex.split(command)
        cmd.extend(command)

        cmd_string = shlex.join(cmd)
        env_string = " ".join([f"{k}={v}" for k, v in env.items()])
        self.__logger.debug(
            "Executing command: " + env_string + " " + cmd_string
        )

        try:
            proc = subprocess.run(
                cmd, capture_output=True, check=True, env=env, text=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                "Command '{}' failed with return code {} "
                "and stderr:\n{}".format(
                    shlex.join(e.cmd), e.returncode, e.stderr
                )
            ) from None

        if self.log_output:
            self.__logger.info(proc.stdout)


class CondorApptainerTask(ApptainerTask):
    submit_dir = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def name(self):
        return "aframe"

    @property
    def queue(self):
        # to allow e.g. "queue start,stop from segments.txt" syntax.
        # this will require a pycondor change
        # to allow `queue` values that are strings
        return "queue"

    def run(self):
        env = ""
        for key, value in self.environment.items():
            env += f"APPTAINERENV_{key} = {value} "

        cmd = ["exec"]
        for source, dest in self._binds.items():
            cmd.extend["--bind", f"{source}:{dest}"]

        # I think we'll only ever be using condor for data generation
        # where gpus are not needed so no need to add that capability
        # here, at least right now.

        cmd.append(self.image)
        command = dedent(self.command).replace("\n", " ")
        command = shlex.split(command)
        cmd.extend(command)

        job = Job(
            name=self.name,
            executable=shutil.which("apptainer"),
            error=self.submit_dir,
            output=self.submit_dir,
            log=self.submit_dir,
            arguments=" ".join(cmd),
            extra_lines=[f"environment = {env}"],
            queue=self.queue,
        )
        # replace with build_submit
        job.build_submit(fancyname=False)


class AframeApptainerTask(ApptainerTask):
    dev = luigi.BoolParameter()

    def __init__(self, *args, **kwargs):
        root = Path(__file__).resolve()
        while root.name != "aframe":
            root = root.parent
        self.root = root.parent
        super().__init__(*args, **kwargs)

        if self.dev:
            self._binds[self.root] = "/opt/aframe"
