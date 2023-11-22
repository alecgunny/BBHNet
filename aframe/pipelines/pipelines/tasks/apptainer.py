import logging
import os
import shlex
import subprocess
from pathlib import Path
from textwrap import dedent
from typing import Sequence

import luigi
import kr8s
from ray_kube import KubernetesRayCluster

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


class K8ServiceTask(AframeApptainerTask):
    container = luigi.Parameter()
    workers = luigi.IntParameter()
    gpus_per_worker = luigi.IntParameter()
    k8s_config = luigi.Parameter(default="~/.kube/config")

    def run(self):
        kr8s.api(kubeconfig=self.k8s_config)
        cluster = KubernetesRayCluster(
            image=self.container,
            num_workers=self.workers,
            gpus_per_worker=self.gpus_per_worker,
        )
        with cluster as cluster:
            self.ip = cluster.get_load_balancer_ip()
            super().run()
