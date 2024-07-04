"""Visualize memory consumpion."""

from os import path

from matplotlib import pyplot as plt
from pandas import read_csv
from tueplots import bundles

HERE = path.abspath(__file__)
HEREDIR = path.dirname(HERE)

DATADIR = path.join(HEREDIR, "gathered")

requires_grads = ["all", "none", "4+", "4"]
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
architectures = ["linear", "conv", "norm_eval"]
architectures = ["norm_eval"]
architectures = ["conv"]

for arch in architectures:
    with plt.rc_context(bundles.icml2024()):
        plt.rcParams.update({"figure.figsize": (3.25, 2.5)})
        fig, ax = plt.subplots()
        ax.set_xlabel("Number of layers")
        ax.set_ylabel("Peak memory [MiB]")

        markerstyle = {"markersize": 3.5, "fillstyle": "none"}

        # visualize PyTorch's behavior
        implementation = "torch"

        for requires_grad in requires_grads:
            df = read_csv(
                path.join(
                    DATADIR,
                    f"peakmem_implementation_{arch}_{implementation}_requires_grad_{requires_grad}.csv",
                )
            )
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
        df = read_csv(
            path.join(
                DATADIR,
                f"peakmem_implementation_{arch}_{implementation}_requires_grad_{requires_grad}.csv",
            )
        )
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
            path.join(HEREDIR, f"visual_abstract_{arch}.pdf"), bbox_inches="tight"
        )
