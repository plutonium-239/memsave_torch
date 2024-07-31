"""Visualize memory consumpion."""

from itertools import product
from os import path

from matplotlib import pyplot as plt
from pandas import read_csv
from tueplots import bundles

HEREDIR = path.dirname(path.abspath(__file__))
DATADIR = path.join(HEREDIR, "gathered")

requires_grads = {"all", "none", "4+", "4"}
legend_entries = {
    "all": "Fully differentiable",
    "none": "Fully non-differentiable",
    "4+": "Layers 4+ differentiable",
    "4": "Layer 4 differentiable",
    "4 (ours)": "Layer 4 differentiable (ours)",
}
markers = {
    "all": "o",
    "none": "x",
    "4+": "<",
    "4": ">",
    "4 (ours)": "p",
}
linestyles = {
    "all": "-",
    "none": "-",
    "4+": "dashed",
    "4": "dashdot",
    "4 (ours)": "dotted",
}
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
modes = {"train", "eval"}
use_compiles = {False, True}

if __name__ == "__main__":
    for architecture, mode, use_compile in product(architectures, modes, use_compiles):
        with plt.rc_context(bundles.icml2024()):
            fig, ax = plt.subplots()
            ax.set_xlabel("Number of layers")
            ax.set_ylabel("Peak memory [MiB]")

            markerstyle = {"markersize": 3.5, "fillstyle": "none"}

            # visualize PyTorch's behavior
            implementation = "torch"

            for requires_grad in requires_grads:
                readpath = path.join(
                    DATADIR,
                    f"peakmem_{architecture}_mode_{mode}_implementation_{implementation}"
                    + f"_requires_grad_{requires_grad}"
                    + f"{'_use_compile' if use_compile else ''}.csv",
                )
                df = read_csv(readpath)
                ax.plot(
                    df["num_layers"],
                    df["peakmem"],
                    label=legend_entries[requires_grad],
                    marker=markers[requires_grad],
                    linestyle=linestyles[requires_grad],
                    **markerstyle,
                )

            # visualize our layer's behavior
            implementation, requires_grad = "ours", "4"
            key = f"{requires_grad} ({implementation})"
            readpath = path.join(
                DATADIR,
                f"peakmem_{architecture}_mode_{mode}_implementation_{implementation}"
                + f"_requires_grad_{requires_grad}"
                + f"{'_use_compile' if use_compile else ''}.csv",
            )
            if path.exists(readpath):
                # for skipping memsave+compiled
                df = read_csv(readpath)
                ax.plot(
                    df["num_layers"],
                    df["peakmem"],
                    label=legend_entries[key],
                    marker=markers[key],
                    linestyle=linestyles[key],
                    **markerstyle,
                )

            plt.legend()
            plt.savefig(
                path.join(HEREDIR, f"visual_abstract_{architecture}_{mode}{'_use_compile' if use_compile else ''}.pdf"),
                bbox_inches="tight",
            )
