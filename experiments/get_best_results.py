"""Simple script that goes over the raw results and finds the best results."""

import argparse
import os
from glob import glob
from itertools import product

import pandas as pd

from experiments.util.collect_results import case_inv_mapping


def main(base_dir: str):
    """Gets the best results from all previous runs

    Args:
        base_dir (str): The base results dir
    """
    # Don't recognize None as NaN
    custom_na_values = pd._libs.parsers.STR_NA_VALUES - {"None"}
    for device, arch in product(["cuda", "cpu"], ["linear", "conv", "transformer"]):
        # usage stats
        df = None
        idx_col = ["model", "case"]
        for fname in glob(os.path.join(base_dir, f"usage_stats-{arch}-{device}-*.csv")):
            with open(fname) as f:
                # f.readline()
                temp_df = pd.read_csv(
                    f,
                    index_col=idx_col,
                    header=1,
                    na_values=custom_na_values,
                    keep_default_na=False,
                )
            df = temp_df if df is None else pd.concat([df, temp_df])
        if df is not None:
            df = df.rename(index=case_inv_mapping, level=1)
            df["Memory Usage (GB)"] = df["Memory Usage (MB)"] / 1024
            df = df.drop(columns=["Memory Usage (MB)"])
            best_results = df.groupby(idx_col).min()
            # scale
            maxes = best_results.groupby(["model"]).max()
            best_results[["Scaled T", "Scaled M"]] = best_results / maxes
            best_results.to_csv(
                os.path.join(base_dir, f"best_results-{arch}-{device}-usage_stats.csv")
            )

        # savings
        df = None
        idx_col = ["model", "input_vjps"]
        for fname in glob(os.path.join(base_dir, f"savings-{arch}-{device}*.csv")):
            with open(fname) as f:
                f.readline()
                temp_df = pd.read_csv(f, index_col=idx_col)
            df = temp_df if df is None else pd.concat([df, temp_df])

        if df is not None:
            best_results = df.groupby(idx_col).max()
            best_results.to_csv(
                os.path.join(base_dir, f"best_results-{arch}-{device}-savings.csv")
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir", type=str, default="results/", help="The base results dir"
    )
    args = parser.parse_args()

    base_dir = args.results_dir
    os.path.exists(base_dir)

    main(base_dir)
