import shlex
import subprocess
from time import sleep
import itertools
from tqdm import tqdm

from convrad_exp.exp01_estimated_speedup import collect_results

estimators = ["time", "memory"]
# estimators = ["memory"]

# improvements can be either speedups or savings based on context
vjp_improvements = [
    0.9,  # doable
    0.5,  # realistic
    0.1,  # ambitious
    0.01,  # unrealistic
]

# CONV
# Valid choices for models are in models.conv_model_fns
# models = ["deepmodel", "alexnet", "resnet101", "resnet18", "vgg16"]  #, "convnext_base"]
models = ["resnet101"]  # "convnext_base"
# models = ['vgg16']
models = [[m, f"memsave_{m}"] for m in models]  # add memsave versions for each model
models = list(itertools.chain.from_iterable(models))  # flatten list of lists
# models = ["memsave_resnet101"]
batch_size = 64
input_channels = 3
input_HW = 224
num_classes = 1000
device = "cuda"
architecture = "conv"

# LINEAR
# Valid choices for models are in models.linear_model_fns
# models = ['deeplinearmodel']
# models += [f"memsave_{m}" for m in models]  # add memsave versions for each model
# batch_size = 32768
# input_channels = 3
# input_HW = 64
# num_classes = 1000
# device = 'cuda'
# architecture = 'linear' # use high batch size

pbar = tqdm(total=len(models) * len(estimators) * 3, leave=False)
collector = collect_results.ResultsCollector(batch_size, input_channels, input_HW, num_classes, device, architecture, vjp_improvements)

for model in models:
    for estimate in estimators:
        outputs = []

        collector.clear_file(estimate)
        for case in [
            None,
            ["grad_input"],
            [
                "grad_input",
                f"no_grad_{architecture}_weights",
                f"no_grad_{architecture}_bias",
            ],
        ]:
            pbar.update()
            pbar.set_description(f"{model} {estimate} case {case}")
            case_str = f"--case {' '.join(case)}" if case is not None else ""
            cmd = (
                f"python estimate.py --architecture {architecture} --model {model} --estimate {estimate} {case_str} "
                + f"--device {device} -B {batch_size} -C_in {input_channels} -HW {input_HW} -n_class {num_classes}"
            )
            proc = subprocess.run(shlex.split(cmd), capture_output=True, shell=True)
            assert proc.stderr in [
                None,
                b"",
            ], f"Error in estimate.py: \n{proc.stderr.decode()}"
            sleep(0.25)

        collector.collect_from_file(estimate, model, case)

pbar.close()
