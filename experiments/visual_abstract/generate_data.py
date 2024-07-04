"""Launch all configurations of the memory benchmark."""

from itertools import product
from os import path
from subprocess import CalledProcessError, run
from typing import List

HERE = path.abspath(__file__)
HEREDIR = path.dirname(HERE)
SCRIPT = path.join(HEREDIR, "run.py")


max_num_layers = 10
requires_grads = ["all", "none", "4", "4+"]
# requires_grads = ["4+"]
implementations = ["torch", "ours"]
# implementations = ["ours"]
architectures = ["linear", "conv", "norm_eval"]
architectures = ["norm_eval"]
architectures = ["linear"]


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
    for implementation, requires_grad, arch in product(
        implementations, requires_grads, architectures
    ):
        if implementation == "ours" and requires_grad != "4":
            continue
        for num_layers in range(1, max_num_layers + 1):
            _run(
                [
                    "python",
                    SCRIPT,
                    f"--implementation={implementation}",
                    f"--architecture={arch}",
                    f"--num_layers={num_layers}",
                    f"--requires_grad={requires_grad}",
                ]
            )
