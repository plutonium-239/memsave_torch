"""Combine data from individual runs into data frames."""

from itertools import product
from os import makedirs, path

from pandas import DataFrame

HERE = path.abspath(__file__)
HEREDIR = path.dirname(HERE)
RAWDATADIR = path.join(HEREDIR, "raw")
DATADIR = path.join(HEREDIR, "gathered")
makedirs(RAWDATADIR, exist_ok=True)
makedirs(DATADIR, exist_ok=True)

max_num_layers = 10
requires_grads = ["all", "none", "4", "4+"]
implementations = ["torch", "ours"]

if __name__ == "__main__":
    for implementation, requires_grad in product(implementations, requires_grads):
        if implementation == "ours" and requires_grad != "4":
            continue

        layers = list(range(1, max_num_layers + 1))
        peakmems = []
        for num_layers in layers:
            with open(
                path.join(
                    RAWDATADIR,
                    f"peakmem_implementation_{implementation}_num_layers_{num_layers}_requires_grad_{requires_grad}.txt",
                ),
                "r",
            ) as f:
                peakmems.append(float(f.read()))

        df = DataFrame({"num_layers": layers, "peakmem": peakmems})
        savepath = path.join(
            DATADIR,
            f"peakmem_implementation_{implementation}_requires_grad_{requires_grad}.csv",
        )
        df.to_csv(savepath, index=False)
