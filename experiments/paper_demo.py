"""Paper Demo

This script replicates all the results given in the paper.
Make sure to run the `get_best_results.py` script.
"""

import shlex
import subprocess
from time import sleep

from tqdm import tqdm

from experiments.util import collect_results
from experiments.util.models import prefix_in_pairs

estimators = ["time", "memory"]
# estimators = ["memory"]
# estimators = ["time"]

# improvements can be either speedups or savings based on context
vjp_improvements = [
    0.9,  # doable
    0.5,  # realistic
    0.1,  # ambitious
    0.01,  # unrealistic
]

# repeat the experiment multiple times (generates multiple files to be aggregated by `get_best_results`)
n_repeat = 5

# ============== CONV CONFIG ==============
# Valid choices for models are in models.conv_model_fns
# models = [
#     "deepmodel",
#     "resnet101",
#     "resnet18",
#     "vgg16",  # "convnext_base",
#     "fasterrcnn_resnet50_fpn_v2",
#     "ssdlite320_mobilenet_v3_large",  # "retinanet_resnet50_fpn_v2",
#     "deeplabv3_resnet101",
#     "fcn_resnet101",
#     "efficientnet_v2_l",
#     "mobilenet_v3_large",
#     "resnext101_64x4d",
# ]
# models = prefix_in_pairs("memsave_", models)
# batch_size = 64
# input_channels = 3
# input_HW = 224
# num_classes = 1000
# device = "cuda"
# architecture = "conv"
# cases = collect_results.select_cases(['All', 'Input', 'Conv', 'Norm'])

# ============== TRANSFORMER CONFIG ==============
# Valid choices for models are in models.transformer_model_fns
models = [
    "transformer",
    "gpt2",
    "bert",
    "bart",
    "roberta",
    "t5",
    "flan-t5",
    # "xlm-roberta",
    "mistral-7b",
    "llama3-8b",
    "phi3-4b",
]
models = prefix_in_pairs("memsave_", models)
batch_size = 64
input_channels = 2048
input_HW = 256
num_classes = 5000
device = "cuda"
architecture = "transformer"
cases = collect_results.select_cases(["All", "Input", "Norm"])

# ============== LINEAR CONFIG ==============
# Valid choices for models are in models.linear_model_fns
# models = ['deeplinearmodel']
# models += [f"memsave_{m}" for m in models]  # add memsave versions for each model
# batch_size = 32768
# input_channels = 3
# input_HW = 64
# num_classes = 1000
# device = 'cuda'
# architecture = 'linear' # use high batch size
# cases = collect_results.select_cases(['All', 'Input', 'Linear'])


if __name__ == "__main__":
    for i_repeat in range(n_repeat):
        print(f" Repetition #{i_repeat} ".center(80, "-"))
        pbar = tqdm(total=len(models) * len(estimators) * len(cases), leave=False)
        collector = collect_results.ResultsCollector(
            batch_size,
            input_channels,
            input_HW,
            num_classes,
            device,
            architecture,
            vjp_improvements,
            cases,
            "results",
        )

        for model in models:
            B = batch_size
            if model in prefix_in_pairs("memsave_", ["flan-t5"]):
                B = 56
            if model in prefix_in_pairs("memsave_", ["mistral-7b", "phi3-4b"]):
                B = 16
            if model in prefix_in_pairs("memsave_", ["llama3-8b"]):
                B = 8
            for estimate in estimators:
                outputs = []

                collector.clear_file(estimate)
                for case in cases:
                    pbar.update()
                    case_display = collect_results.case_inv_mapping[
                        collect_results.make_case_str(case)
                    ]
                    case_str = f"--case {' '.join(case)}" if case is not None else ""
                    pbar.set_description(f"{model} {estimate} case {case_display}")
                    cmd = (
                        f"python experiments/util/estimate.py --architecture {architecture} --model {model} --estimate {estimate} {case_str} "
                        + f"--device {device} -B {B} -C_in {input_channels} -HW {input_HW} -n_class {num_classes}"
                    )
                    proc = subprocess.run(shlex.split(cmd), capture_output=True)
                    assert proc.stderr in [
                        None,
                        b"",
                    ], f"Error in estimate.py: \n{proc.stderr.decode()}"
                    sleep(0.25)

                collector.collect_from_file(estimate, model)

        collector.finish()
        pbar.close()
