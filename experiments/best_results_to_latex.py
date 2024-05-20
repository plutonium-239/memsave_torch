"""Simple script to make a latex table from best results"""

import pandas as pd

df = pd.read_csv("results/llm/best_results-transformer-cuda-usage_stats.csv")

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


def _format_name(n):
    if n.startswith("memsave_"):
        mname = n.split("memsave_", 1)[1]
        return f"{mname.capitalize()} + MemSave"
    return n.capitalize()


ni = df2["model"].apply(_format_name)
df2 = (
    df2.set_index(ni)
    .sort_index()
    .drop(
        columns=[
            "model",
            "memsave",
            "Memory Usage (GB)",
            "Time Taken (s)",
            "Scaled M",
            "Scaled T",
        ]
    )
)

df2_p = df2.pivot_table(
    index="model", columns="case", values=df2.columns[1:], aggfunc=lambda x: x
)

print(df2_p.to_latex(na_rep="-"))
