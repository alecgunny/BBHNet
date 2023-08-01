import argparse
import logging
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import toml
from numpy.random import default_rng

source_dir = Path(__file__).resolve().parent


def configure_logger(filename):
    logger = logging.getLogger()
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        format=log_format,
        level=logging.INFO,
        stream=sys.stdout,
    )

    logger = logging.getLogger()
    if filename is not None:
        formatter = logging.Formatter(log_format)
        handler = logging.FileHandler(filename=filename, mode="w")
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def create_random_hp_sets(n_iter: int = 10):
    rng = default_rng()
    hp_set = {}
    hp_set["iteration"] = np.arange(1, n_iter + 1)

    waveform_probs = rng.uniform(0.3, 0.7, n_iter)
    probs = np.ones(n_iter)
    swap_frac = np.zeros(n_iter)
    mute_frac = np.zeros(n_iter)
    mask = probs >= 1
    while mask.any():
        swaps = rng.uniform(0, 0.15, mask.sum())
        swap_frac[mask] = swaps

        mutes = rng.uniform(0, 0.3, mask.sum())
        mute_frac[mask] = mutes

        downweight = 1 - (swap_frac + mute_frac - (mute_frac * swap_frac))
        probs = waveform_probs / downweight
        mask = probs >= 1

    hp_set["waveform_prob"] = waveform_probs
    hp_set["swap_frac"] = swap_frac
    hp_set["mute_frac"] = mute_frac
    hp_set["snr_decay_steps"] = rng.integers(1, 2500, n_iter)
    hp_set["max_lr"] = 10 ** (rng.uniform(-4.5, -2, n_iter))
    hp_set["lr_ramp_epochs"] = rng.integers(2, 50, n_iter)

    return hp_set


class HpSearch:
    def __init__(self, results_dir: Path, n_iter: int = 10):
        self.logger = logging.getLogger(results_dir.name)
        self.n_iter = n_iter
        self.results_dir = results_dir
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.hyperparameters = create_random_hp_sets(n_iter)

    @property
    def config_path(self):
        return self.results_dir / "pyproject.toml"

    @property
    def dotenv_path(self):
        return self.results_dir / "search.env"

    def read_config(self):
        with open(self.config_path, "r") as f:
            return toml.load(f)

    def update_config(self, config, hyperparameters):
        # infer the name of the output directory
        iteration = hyperparameters.pop("iteration")
        outdir = f"iteration_{iteration}"
        outdir = self.results_dir / outdir
        outdir.mkdir(exist_ok=True, parents=False)

        # assign the output directory via the environment
        # file's BASE_DIR variable
        with open(self.dotenv_path, "r") as f:
            env = f.read()
        if "BASE_DIR" not in env:
            env += f"BASE_DIR={outdir}\n"
        else:
            env = re.sub("(?m)(?<=BASE_DIR=).+", str(outdir), env)
        with open(self.dotenv_path, "w") as f:
            f.write(env)

        # update all the relevant portions of the config
        typeo = config["tool"]["typeo"]
        for hp, value in hyperparameters.items():
            if hp == "norm_groups":
                typeo["base"]["resnet"][hp] = value
            else:
                typeo["scripts"]["train"][hp] = value

        # write the config to the run's directory
        # and return the path to point our train
        # command at it
        config_path = outdir / "pyproject.toml"
        with open(config_path, "w") as f:
            toml.dump(config, f)
        return config_path

    def iter_hps(self):
        for i in range(self.n_iter):
            yield {k: v[i] for k, v in self.hyperparameters.items()}

    def launch(self):
        """
        Train over sets of hyperparameters
        """
        with h5py.File(self.results_dir / "params.h5", "w") as f:
            for k, v in self.hyperparameters.items():
                f.create_dataset(k, data=v)

        config = self.read_config()
        for param_set in self.iter_hps():
            # Update config with this set of hyperparameters
            config_path = self.update_config(config, param_set)
            param_str = " ".join(
                [f"{k}: {v:g}," for k, v in param_set.items()]
            )
            self.logger.info("Beginning with parameter set " + param_str[:-1])

            cmd = [
                "pinto",
                "-p",
                str(source_dir / "train"),
                "run",
                "-e",
                str(self.dotenv_path),
                "train",
                "--typeo",
                f"{config_path}:train:resnet",
            ]
            self.logger.info(" ".join(cmd))
            subprocess.check_output(cmd)


def ignore(dirname, contents):
    if dirname.endswith(".git"):
        return contents
    elif dirname.endswith("notebook"):
        return contents
    else:
        return []


def launch_search(results_dir, gpu, N):
    # create a search, which will create its associated
    # directory, then copy the source config and
    # environment file into its root location
    search = HpSearch(results_dir / f"GPU-{gpu}", N)
    shutil.copy(source_dir / "pyproject.toml", search.config_path)
    shutil.copy(source_dir / "search.env", search.dotenv_path)

    with open(search.dotenv_path, "r") as f:
        env = f.read()
    env += f"CUDA_VISIBLE_DEVICES={gpu}\n"
    with open(search.dotenv_path, "w") as f:
        f.write(env)

    logging.info(f"Launching hyperparameter sweep on GPU {gpu}")
    search.launch()

    return gpu


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name", type=str)
    parser.add_argument("--gpus", type=int, nargs="+")
    parser.add_argument("-N", type=int, default=10)
    args = parser.parse_args()

    results_dir = Path.home() / "aframe" / args.run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    configure_logger(results_dir / "search.log")

    futures = []
    with ThreadPoolExecutor(len(args.gpus)) as ex:
        for gpu in args.gpus:
            future = ex.submit(launch_search, results_dir, gpu, args.N)
            futures.append(future)

    for f in as_completed(futures):
        gpu = f.result()
        logging.info(f"Finished search on GPU {gpu}")
