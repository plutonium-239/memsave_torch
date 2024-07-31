"""Script to plot bar graphs of savings as seen in the poster"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tueplots import bundles

df = pd.read_csv("results/paper/poster.csv")
df["Scaled M str"] = df["Scaled M"].apply(lambda x: f"{x:.2f}")
df["M str"] = df["Memory Usage (GB)"].apply(lambda x: f"{x:.2f}")

memsave_map = {False: "PyTorch", True: "+ MemSave"}
df["colors"] = df["memsave"].apply(lambda x: memsave_map[x])
color_map = {memsave_map[False]: "#F05F42", memsave_map[True]: "#00E1D2"}

# fig = px.bar(df, x='case', y='Scaled M', color='colors', text='M str',
#     category_orders={'case': ['All', 'Input', 'Norm', 'SurgicalFirst']},
#     barmode='group', facet_col='model_clean', facet_col_wrap=3,
#     color_discrete_map={memsave_map[False]: '#F05F42', memsave_map[True]: '#00E1D2'},
# )

# fig.update_traces(width=0.6)
# fig.show()

width = 0.4
df["color_val"] = df["colors"].apply(lambda x: color_map[x])

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
    # Conv
    "resnet101": "ResNet-101",
    "deeplabv3_resnet101": "DeepLabv3 (RN101)",
    "efficientnet_v2_l": "EfficientNetv2-L",
    "fcn_resnet101": "FCN (RN101)",
    "mobilenet_v3_large": "MobileNetv3-L",
    "resnext101_64x4d": "ResNeXt101-64x4d",
    "fasterrcnn_resnet50_fpn_v2": "Faster-RCNN (RN101)",
    "ssdlite320_mobilenet_v3_large": "SSDLite (MobileNetv3-L)",
    "vgg16": "VGG-16",
}

for chosen_model in ["resnet101", "efficientnet_v2_l", "mistral-7b", "t5"]:
    df_model = df[df["model_clean"] == chosen_model]
    with plt.rc_context(bundles.icml2024(column="full")):
        fig, ax = plt.subplots()
        # ax.set_xlabel("Case", size='large')
        ax.set_ylabel("Peak memory [GiB]", size="large")
        cases = []
        for i, (case, group) in enumerate(df_model.groupby("case")):
            cases.append(case)
            for j, (memsave, mg) in enumerate(group.groupby("memsave")):  # noqa: B007
                r = ax.bar(
                    i + j * width,
                    mg["Memory Usage (GB)"],
                    width,
                    label=mg["colors"].item(),
                    color=mg["color_val"],
                )
                ax.bar_label(r, mg["Scaled M str"], padding=-20, size="x-large")
                yoff = mg["Memory Usage (GB)"].item() * 0.05
                if r[0].get_height() < 5:
                    ax.text(
                        i + j * width,
                        r[0].get_height() + yoff,
                        mg["colors"].item(),
                        ha="center",
                        va="bottom",
                        rotation="vertical",
                        size="x-large",
                    )
                else:
                    ax.text(
                        i + j * width,
                        yoff,
                        mg["colors"].item(),
                        ha="center",
                        va="bottom",
                        rotation="vertical",
                        size="x-large",
                    )

                # ax.bar(i + width, group['Scaled M'], width, label=group['M str'])
            # ax.bar_label(rects, padding=3)

            # for memsave, sub_group in group.groupby('memsave'):
            #     ax.plot(sub_group['case'], sub_group['Memory Usage (GB)'], marker='o', linestyle=linestyle, color=color, label=f'{model_clean} - {"memsave" if memsave else "no memsave"}')
            #     for j, txt in enumerate(sub_group['Scaled M str']):
            #         ax.annotate(txt, (sub_group['case'].iloc[j], sub_group['Memory Usage (GB)'].iloc[j]))

        ax.set_xticks(np.arange(len(cases)) + width / 2, cases)
        ax.tick_params(labelsize="x-large")
        ax.set_title(names[chosen_model], fontsize="xx-large", fontweight=1000)
        # handles, labels = ax.get_legend_handles_labels()
        # unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        # ax.legend(*zip(*unique))

        # ax.legend()
        # fig.show()
        # fig.waitforbuttonpress()
        plt.savefig(
            f"results/paper/poster_plot_{chosen_model}.pdf",
            bbox_inches="tight",
        )
