"""Combine data from individual runs into data frames."""

from itertools import product
from os import makedirs, path

from pandas import DataFrame

HEREDIR = path.dirname(path.abspath(__file__))
RAWDATADIR = path.join(HEREDIR, "raw")
DATADIR = path.join(HEREDIR, "gathered")
makedirs(RAWDATADIR, exist_ok=True)
makedirs(DATADIR, exist_ok=True)

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

if __name__ == "__main__":
    for implementation, requires_grad, architecture, mode, use_compile in product(
        implementations, requires_grads, architectures, modes, use_compiles
    ):
        if implementation == "ours" and requires_grad != "4":
            continue

        peakmems = []
        layers = list(range(1, max_num_layers + 1))
        for num_layers in layers:
            readpath = path.join(
                RAWDATADIR,
                f"peakmem_{architecture}_mode_{mode}_implementation_{implementation}"
                + f"_num_layers_{num_layers}_requires_grad_{requires_grad}"
                f"{'_use_compile' if use_compile else ''}.txt",
            )
            with open(readpath, "r") as f:
                peakmems.append(float(f.read()))

        df = DataFrame({"num_layers": layers, "peakmem": peakmems})
        savepath = path.join(
            DATADIR,
            f"peakmem_{architecture}_mode_{mode}_implementation_{implementation}"
            + f"_requires_grad_{requires_grad}{'_use_compile' if use_compile else ''}"
            + ".csv",
        )
        df.to_csv(savepath, index=False)
