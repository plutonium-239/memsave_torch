"""Simple script to make a latex table from resnet results"""

import pandas as pd

df = pd.read_csv("results/resnet101_only/best_results-conv-cpu-usage_stats.csv")
df = df.set_index("model")
df = df.drop(columns=["Scaled M", "Scaled T"])
df = df.drop("memsave_resnet101_conv+relu+bn")
df = df[df["case"] != "SurgicalLast"]
df = df[df["case"] != "Conv"]

mem_div = df[df["case"] == "All"].loc["resnet101", "Memory Usage (GB)"]
time_div = df[df["case"] == "All"].loc["resnet101", "Time Taken (s)"]
df["Scaled M"] = df["Memory Usage (GB)"] / mem_div
df["Scaled T"] = df["Time Taken (s)"] / time_div

df["Memory [GiB]"] = df.apply(
    lambda x: f"{x['Memory Usage (GB)']:.2f} ({x['Scaled M']:.2f})", axis=1
)
df["Time [s]"] = df.apply(
    lambda x: f"{x['Time Taken (s)']:.2f} ({x['Scaled T']:.2f})", axis=1
)

df = df.drop(columns=["Scaled M", "Scaled T", "Memory Usage (GB)", "Time Taken (s)"])
df_p = df.pivot_table(
    index="model", columns="case", values=df.columns[1:], aggfunc=lambda x: x
)

labels = {
    "resnet101": "Default ResNet-101",
    "memsave_resnet101_conv": "+ swap Convolution",
    "memsave_resnet101_conv_full": "+ swap BatchNorm, ReLU",
}

df_p = df_p.rename(index=labels)
df_p = df_p.sort_index(ascending=False)

print(df_p["Memory [GiB]"].to_latex())
print(df_p["Time [s]"].to_latex())
