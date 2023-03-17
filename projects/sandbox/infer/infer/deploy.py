import math
import re
import shutil
import socket
import subprocess
from pathlib import Path
from textwrap import dedent
from typing import List

from typeo import scriptify

from hermes.aeriel.serve import serve

re_dagman_cluster = re.compile(r"(?<=submitted\sto\scluster )[0-9]+")
re_fname = re.compile(r"([0-9]{10})-([1-9][0-9]*)\.")


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
        match = re_fname.search(fname)
        if match is not None:
            duration = match.group(1)
            T += duration
    return calc_shifts_required(Tb, T, shift)


def get_exectubale(name: str) -> str:
    ex = shutil.which(name)
    if ex is None:
        raise ValueError(f"No executable {ex}")
    return str(ex)


def get_ip_address() -> str:
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    return ip


def create_submit_file(
    executable: str,
    condor_dir: Path,
    accounting_group: str,
    accounting_group_user: str,
    arguments: str,
):
    logdir = condor_dir / "logs"
    logdir.mkdir(exist_ok=True, parents=True)
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
        request_memory = 1024
        request_disk = 1024
        queue shifts,seq_id from {condor_dir}/shifts.txt
    """
    )
    return subfile


@scriptify
def main(
    model_name: str,
    model_repo: str,
    image: str,
    output_dir: Path,
    data_dir: Path,
    accounting_group: str,
    accounting_group_user: str,
    Tb: float,
    shift: float,
    injection_set_file: Path,
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
    # get ip address and add to arguments
    # along with the timeslide datadir which will be read from a text file
    ip = get_ip_address()
    log_pattern = "infer-$(ProcID).log"
    output_pattern = "tmp/output-$(ProcID)"
    arguments = f"""
    --data-dir ${data_dir}
    --output_dir {output_dir / output_pattern}
    --shifts $(shifts)
    --sequence-id $(seq_id)
    --log-file {output_dir / log_pattern}
    --ip {ip}
    --model-name {model_name}
    --injection-set-file {injection_set_file}
    --sample-rate {sample_rate}
    --inference-sampling-rate {inference_sampling_rate}
    --ifos {ifos}
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
    executable = shutil.which("infer")
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

    num_shifts = get_num_shifts(data_dir, Tb)
    with open(condor_dir / "shifts.txt", "w") as f:
        for i in range(num_shifts):
            seq_id = sequence_id + 2 * i
            f.write(f"0 {i * shift},{seq_id}")

    condor_submit = shutil.which("condor_submit")
    if condor_submit is None:
        raise ValueError("No condor_submit executable")

    cmd = [condor_submit, str(subfile_path)]

    # spin up triton server
    with serve(model_repo, image, wait=True):
        # launch inference jobs via condor
        out = subprocess.check_output(cmd, text=True)

        # find dagman id and wait for jobs to finish
        dagid = int(re_dagman_cluster.search(out).group())
        cwq = shutil.which("condor_watch_q")
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


if __name__ == "__main__":
    main()
