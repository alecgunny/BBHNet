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
from pycondor.cluster import JobStatus

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

    @property
    def base_command(self):
        return ["apptainer", "run"]

    def build_command(self):
        cmd = self.base_command
        for source, dest in self._binds.items():
            cmd.extend(["--bind", f"{source}:{dest}"])

        if self.gpus:
            cmd.append("--nv")
            gpus = ",".join(map(str, self.gpus))
            cmd.extend(["--env", f"APPTAINERENV_CUDA_VISIBLE_DEVICES={gpus}"])

        cmd.append(self.image)
        self.__logger.debug(self.command)
        command = dedent(self.command).replace("\n", " ")
        command = shlex.split(command)
        cmd.extend(command)

        return cmd

    def build_env(self):
        env = {}
        for key, value in self.environment.items():
            env[f"APPTAINERENV_{key}"] = value
        return env

    def run(self):
        env = self.build_env()
        cmd = self.build_command()

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


class AframeApptainerTask(ApptainerTask):
    dev = luigi.BoolParameter(default=False)

    def __init__(self, *args, **kwargs):
        base = Path(__file__).resolve()
        while base.name != "aframe":
            base = base.parent
        self.base = base.parent
        super().__init__(*args, **kwargs)

        if self.dev:
            self._binds[self.base] = "/opt/aframe"


class CondorApptainerTask(AframeApptainerTask):
    submit_dir = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__logger = logger

    @property
    def name(self):
        return "aframe"

    @property
    def base_command(self):
        return ["run"]

    @property
    def job_kwargs(self):
        return {}

    @property
    def binds(self):
        # bind for ease of data discovery
        return {"/cvmfs": "/cvmfs"}

    @property
    def environment(self) -> dict:
        return {
            "GWDATAFIND_SERVER": os.getenv("GWDATAFIND_SERVER", ""),
            "BEARER_TOKEN_FILE": "$$(CondorScratchDir)/.condor_creds/igwn.use",
        }

    @property
    def requirements(self):
        return "HasSingularity"

    @property
    def scitoken_lines(self):
        return [
            "use_oauth_services = igwn",
            "igwn_oauth_permissions = read:/ligo,read:/virgo,gwdatafind.read",
            "igwn_oauth_resource = ANY",
        ]

    @property
    def queue(self):
        return None

    def build_env(self):
        env = ""
        for key, value in self.environment.items():
            env += f" APPTAINERENV_{key}={value}"
        return f"{env}"

    def run(self):
        env = self.build_env()
        cmd = self.build_command()
        extra_lines = [f'environment = "{env}"']
        extra_lines.extend(self.scitoken_lines)

        job_kwargs = {
            "name": self.name,
            "submit_name": self.name,
            "error": self.submit_dir,
            "output": self.submit_dir,
            "log": self.submit_dir,
            "submit": self.submit_dir,
            "requirements": self.requirements,
            "extra_lines": extra_lines,
        }

        job_kwargs.update(self.job_kwargs)
        self.__logger.debug(job_kwargs)
        job = Job(
            executable=shutil.which("apptainer"),
            arguments=" ".join(cmd),
            queue=self.queue,
            **job_kwargs,
        )

        job.build(fancyname=False)
        self.__logger.debug(job.submit_file)
        cluster = job.submit_job()
        # wait for all the jobs to finish:
        while not cluster.check_status(JobStatus.COMPLETED, how="all"):
            if cluster.check_status(
                [JobStatus.HELD, JobStatus.FAILED, JobStatus.CANCELLED],
                how="any",
            ):
                raise ValueError("Something went wrong!")
