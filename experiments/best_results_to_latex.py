"""Simple script to make a latex table from best results"""

import pandas as pd

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to best_results<params>.csv generated by get_best_results",
)

args = parser.parse_args()

df = pd.read_csv(args.input)

df = df.set_index("model")
df = df[df["case"] != "Conv"]

df["memsave"] = df.index.str.startswith("memsave_")
badi = df.index.map(
    lambda x: x.split("memsave_", 1)[1] if x.startswith("memsave") else x
)
badi.name = "model_clean"
df2 = df.reset_index().set_index(badi).sort_index()
divs = df2[(df2["case"] == "All") & (~df2["memsave"])]
df2["Scaled M"] = df2["Memory Usage (GB)"] / divs["Memory Usage (GB)"]
df2["Scaled T"] = df2["Time Taken (s)"] / divs["Time Taken (s)"]

df2["Memory [GiB]"] = df2.apply(
    lambda x: f"{x['Memory Usage (GB)']:.2f} ({x['Scaled M']:.2f})", axis=1
)
df2["Time [s]"] = df2.apply(
    lambda x: f"{x['Time Taken (s)']:.2f} ({x['Scaled T']:.2f})", axis=1
)


def _highlight(group, col_sort, col_bold):
    for c_s, c_b in zip(col_sort, col_bold):
        min_idx = group[c_s].argmin()
        group[c_b] = [
            f"\\textbf{{{group.iloc[i][c_b]}}}" if i == min_idx else group.iloc[i][c_b]
            for i in range(len(group.index))
        ]
    return group


df2 = df2.groupby(["model_clean", "case"]).apply(
    _highlight, ["Memory Usage (GB)"], ["Memory [GiB]"]
)
# .apply(_highlight, ['Memory Usage (GB)', 'Time Taken (s)'], ['Memory [GiB]', 'Time [s]'])

names = {
    "bert": "BERT",
    "bart": "BART",
    "roberta": "RoBERTa",
    "gpt2": "GPT-2",
    "t5": "T5",
    "flan-t5": "FLAN-T5",
    "mistral-7b": "Mistral-7B",
    "transformer": "Transformer",
    "llama3-8b": "LLaMa3-8B",
    "phi3-4b": "Phi3-4B",
}


def _format_name(n):
    if n.startswith("memsave_"):
        mname = n.split("memsave_", 1)[1]
        return f"{names[mname]} + MemSave"
    return names[n]


ni = df2["model"].apply(_format_name)
df2 = df2.set_index(ni).sort_index().drop(
    columns=[
        "model",
        "memsave",
        "Memory Usage (GB)",
        "Time Taken (s)",
        "Scaled M",
        "Scaled T",
    ]
)  # fmt: skip

df2_p = df2.pivot_table(
    index="model", columns="case", values=df2.columns[1:], aggfunc=lambda x: x
)

short_index = df2_p.index.map(lambda t: "+ MemSave" if "+ MemSave" in t else t)
df2_p = df2_p.set_index(short_index)

print(df2_p.to_latex(na_rep="-", multicolumn_format="c"))
