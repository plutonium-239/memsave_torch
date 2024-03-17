import shlex
import subprocess
from time import sleep
import itertools
from tqdm import tqdm
import torch.nn as nn

import collect_results
from torchvision.models import resnet101


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
models = ["deepmodel", "resnet101", "resnet18", "vgg16", #"convnext_base",
    "fasterrcnn_resnet50_fpn_v2", "ssdlite320_mobilenet_v3_large", #"retinanet_resnet50_fpn_v2", 
    "deeplabv3_resnet101", "fcn_resnet101", 
    "efficientnet_v2_l", "mobilenet_v3_large", "resnext101_64x4d"]

# models = ["resnet101", "memsave_resnet101_conv", "memsave_resnet101_conv+relu+bn", "memsave_resnet101_conv_full"]
# models = ["resnet101", "memsave_resnet101_conv_full"]

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

cases = [
    None,   # ALL
    [       # INPUT
        "grad_input",
        "no_grad_conv_weights",
        "no_grad_conv_bias",
        "no_grad_linear_weights",
        "no_grad_linear_bias",
        "no_grad_norm_weights",
        "no_grad_norm_bias",       
    ],
    [       # CONV
        "no_grad_linear_weights",
        "no_grad_linear_bias",
        "no_grad_norm_weights",
        "no_grad_norm_bias",       
    ],
    [       # NORM
        "no_grad_conv_weights",
        "no_grad_conv_bias",
        "no_grad_linear_weights",
        "no_grad_linear_bias",    
    ],
]

pbar = tqdm(total=len(models) * len(estimators) * 3, leave=False)
collector = collect_results.ResultsCollector(batch_size, input_channels, input_HW, num_classes, device, architecture, vjp_improvements, cases)

for model in models:
    for estimate in estimators:
        outputs = []

        collector.clear_file(estimate)
        for case in cases:
            pbar.update()
            pbar.set_description(f"{model} {estimate} case {case}")
            case_str = f"--case {' '.join(case)}" if case is not None else ""
            cmd = (
                f"python estimate.py --architecture {architecture} --model {model} --estimate {estimate} {case_str} "
                + f"--device {device} -B {batch_size} -C_in {input_channels} -HW {input_HW} -n_class {num_classes}"
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