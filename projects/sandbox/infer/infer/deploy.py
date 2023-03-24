import logging
import math
import re
import shutil
import subprocess
from pathlib import Path
from textwrap import dedent
from typing import List

import psutil
from typeo import scriptify

from bbhnet.analysis.ledger.events import EventSet, RecoveredInjectionSet
from bbhnet.logging import configure_logging
from hermes.aeriel.serve import serve

re_dagman_cluster = re.compile(r"(?<=submitted\sto\scluster )[0-9]+")
re_fname = re.compile(r"([0-9]{10})-([1-9][0-9]*)\.")


def _fix_path():
    # TODO: fix this on the pinto side
    import os

    bin_dir = "/home/alec.gunny/miniconda3/envs/infer-N6p8Fb-q-py3.9/bin"
    os.environ["PATH"] = bin_dir + ":" + os.environ["PATH"]


def aggregate_results(output_directory: Path):
    background, foreground = EventSet(), RecoveredInjectionSet()
    for data_dir in (output_directory / "tmp").iterdir():
        bckground = EventSet.read(data_dir / "background.h5")
        frground = RecoveredInjectionSet.read(data_dir / "foreground.h5")

        background.append(bckground)
        foreground.append(frground)

    background.write(output_directory / "background.h5")
    foreground.write(output_directory / "foreground.h5")
    shutil.rmtree(output_directory / "tmp")


def calc_shifts_required(Tb: float, T: float, delta: float) -> int:
    r"""
    The algebra to get this is gross but straightforward.
    Just solving
    $$\sum_{i=0}^{N-1}(T - i\delta) \geq T_b$$
    for the lowest value of N, where \delta is the
    shift increment.

    TODO: generalize to multiple ifos and negative
    shifts, since e.g. you can in theory get the same
    amount of Tb with fewer shifts if for each shift
    you do its positive and negative. This should just
    amount to adding a factor of 2 * number of ifo
    combinations in front of the sum above.
    """

    discriminant = (T - delta / 2) ** 2 - 2 * delta * Tb
    N = (T + delta / 2 - discriminant**0.5) / delta
    return math.ceil(N)


def get_num_shifts(data_dir: Path, Tb: float, shift: float) -> int:
    T = 0
    for fname in data_dir.iterdir():
        match = re_fname.search(fname.name)
        if match is not None:
            duration = match.group(2)
            T += float(duration)
    return calc_shifts_required(Tb, T, shift)


def get_executable(name: str) -> str:
    ex = shutil.which(name)
    if ex is None:
        raise ValueError(f"No executable {name}")
    return str(ex)


def get_ip_address() -> str:
    nets = psutil.net_if_addrs()
    return nets["enp1s0f0"][0].address


def create_submit_file(
    executable: str,
    condor_dir: Path,
    accounting_group: str,
    accounting_group_user: str,
    arguments: str,
):
    logdir = condor_dir / "logs"
    logdir.mkdir(exist_ok=True, parents=True)

    # clear any existing logs
    for fname in logdir.iterdir():
        fname.unlink()

    subfile = dedent(
        f"""\
        universe = vanilla
        executable = {executable}
        arguments =  {arguments}
        log = {logdir}/infer-$(ProcId).log
        output = {logdir}/infer-$(ProcId).out
        error = {logdir}/infer-$(ProcId).err
        getenv = True
        accounting_group = {accounting_group}
        accounting_group_user = {accounting_group_user}
        request_memory = 6G
        request_disk = 1G
        queue shift0,shift1,seq_id from {condor_dir}/shifts.txt
    """
    )
    return subfile


@scriptify
def main(
    model_repo_dir: str,
    output_dir: Path,
    data_dir: Path,
    injection_set_file: Path,
    image: str,
    model_name: str,
    accounting_group: str,
    accounting_group_user: str,
    Tb: float,
    shift: float,
    sample_rate: float,
    inference_sampling_rate: float,
    ifos: List[str],
    batch_size: int,
    integration_window_length: float,
    cluster_window_length: float,
    fduration: float,
    throughput: float,
    chunk_size: float,
    sequence_id: int,
    model_version: int = -1,
    verbose: bool = False,
):
    _fix_path()  # TODO: replace this
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(output_dir / "infer.deploy.log", verbose)

    # get ip address and add to arguments
    # along with the timeslide datadir which will be read from a text file
    ip = get_ip_address()
    log_pattern = "infer-$(ProcID).log"
    output_pattern = "tmp/output-$(ProcID)"
    arguments = f"""
    --data-dir {data_dir}
    --output-dir {output_dir / output_pattern}
    --shifts $(shift0) $(shift1)
    --sequence-id $(seq_id)
    --log-file {output_dir / log_pattern}
    --ip {ip}
    --model-name {model_name}
    --injection-set-file {injection_set_file}
    --sample-rate {sample_rate}
    --inference-sampling-rate {inference_sampling_rate}
    --ifos {" ".join(ifos)}
    --batch-size {batch_size}
    --integration-window-length {integration_window_length}
    --cluster-window-length {cluster_window_length}
    --fduration {fduration}
    --throughput {throughput}
    --chunk-size {chunk_size}
    --model-version {model_version}
    """
    arguments = dedent(arguments).replace("\n", " ")
    if verbose:
        arguments += " --verbose"

    condor_dir = output_dir / "condor"
    condor_dir.mkdir(exist_ok=True, parents=True)

    executable = get_executable("infer")
    subfile = create_submit_file(
        executable,
        condor_dir,
        accounting_group,
        accounting_group_user,
        arguments,
    )

    # write the submit file
    subfile_path = condor_dir / "infer.submit"
    with open(subfile_path, "w") as f:
        f.write(subfile)

    num_shifts = get_num_shifts(data_dir, Tb, shift)
    logging.info(f"Submitting jobs for {num_shifts} shifts")
    with open(condor_dir / "shifts.txt", "w") as f:
        for i in range(num_shifts):
            seq_id = sequence_id + 2 * i
            f.write(f"0,{i * shift},{seq_id}\n")

    condor_submit = get_executable("condor_submit")
    cwq = get_executable("condor_watch_q")
    cmd = [condor_submit, str(subfile_path)]

    # spin up triton server
    logging.info("Launching triton server")
    with serve(model_repo_dir, image, wait=True):
        # launch inference jobs via condor
        out = subprocess.check_output(cmd, text=True)

        # find dagman id and wait for jobs to finish
        dagid = int(re_dagman_cluster.search(out).group())
        subprocess.check_call(
            [
                cwq,
                "-exit",
                "all,done,0",
                "-exit",
                "any,held,1",
                "-clusters",
                str(dagid),
            ]
        )

    logging.info("Aggregating results")
    aggregate_results(output_dir)


if __name__ == "__main__":
    main()
