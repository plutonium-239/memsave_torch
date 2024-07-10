"""Launch all configurations of the memory benchmark."""

from itertools import product
from os import path
from subprocess import CalledProcessError, run
from typing import List

from tqdm import tqdm

HERE = path.abspath(__file__)
HEREDIR = path.dirname(HERE)
SCRIPT = path.join(HEREDIR, "run.py")

max_num_layers = 10
requires_grads = {"all", "none", "4", "4+"}
implementations = {"torch", "ours"}
architectures = {
    "linear",
    "conv1d",
    "conv2d",
    "conv3d",
    "bn2d",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
}
modes = {"eval", "train"}
use_compiles = {False, True}

skip_existing = True


def _run(cmd: List[str]):
    """Run the command and print the output/stderr if it fails.

    Args:
        cmd: The command to run.

    Raises:
        CalledProcessError: If the command fails.
    """
    try:
        print(f"Running command: {' '.join(cmd)}")
        job = run(cmd, capture_output=True, text=True, check=True)
        print(f"STDOUT:\n{job.stdout}")
        print(f"STDERR:\n{job.stderr}")
    except CalledProcessError as e:
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        raise e


if __name__ == "__main__":
    configs = list(
        product(implementations, requires_grads, architectures, modes, use_compiles)
    )
    for implementation, requires_grad, architecture, mode, use_compile in tqdm(configs):
        if implementation == "ours" and requires_grad != "4":
            continue

        for num_layers in range(1, max_num_layers + 1):
            _run(
                [
                    "python",
                    SCRIPT,
                    f"--implementation={implementation}",
                    f"--architecture={architecture}",
                    f"--num_layers={num_layers}",
                    f"--requires_grad={requires_grad}",
                    f"--mode={mode}",
                ]
                + (["--skip_existing"] if skip_existing else [])
                + (["--use_compile"] if use_compile else []),
            )
