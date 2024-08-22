# ruff: noqa
import argparse

import torch
from torchview import draw_graph
from torchviz import make_dot

from tqdm import tqdm
import memsave_torch
from experiments.util.collect_results import select_cases
from experiments.util.estimate import parse_case
from experiments.util.measurements import separate_grad_arguments


def eval_bn(num_features):
    m = torch.nn.BatchNorm2d(num_features)
    m.eval()
    return m


to_test = [
    {
        "name": "Linear2dims",
        "layer_fn": lambda: torch.nn.Linear(3, 5),
        "data_fn": lambda: torch.rand(7, 12, 3),  # weight sharing
        "grads": ["All", "Input", "Linear", "Everything", "Nothing"],
    },
    {
        "name": "Conv2d",
        "layer_fn": lambda: torch.nn.Conv2d(3, 5, 3),
        "data_fn": lambda: torch.rand(7, 3, 12, 12),
        "grads": ["All", "Input", "Conv", "Everything", "Nothing"],
    },
    {
        "name": "BatchNorm2d",
        "layer_fn": lambda: torch.nn.BatchNorm2d(3),
        "data_fn": lambda: torch.rand(7, 3, 12, 12),
        "grads": ["All", "Input", "Norm", "Everything", "Nothing"],
    },
    {
        "name": "BatchNorm2d_Eval",
        "layer_fn": lambda: eval_bn(3),
        "data_fn": lambda: torch.rand(7, 3, 12, 12),
        "grads": ["All", "Input", "Norm", "Everything", "Nothing"],
    },
    {
        "name": "LayerNorm",
        "layer_fn": lambda: torch.nn.LayerNorm([3, 12, 12]),
        "data_fn": lambda: torch.rand(7, 3, 12, 12),
        "grads": ["All", "Input", "Norm", "Everything", "Nothing"],
    },
    {
        "name": "Dropout",
        "layer_fn": lambda: torch.nn.Dropout(),
        "data_fn": lambda: torch.rand(7, 3, 12, 12),
        "grads": ["All", "Input", "Everything", "Nothing"],
    },
    {
        "name": "MaxPool2d",
        "layer_fn": lambda: torch.nn.MaxPool2d(3),
        "data_fn": lambda: torch.rand(7, 3, 12, 12),
        "grads": ["All", "Input", "Everything", "Nothing"],
    },
    {
        "name": "ReLU",
        "layer_fn": lambda: torch.nn.ReLU(),
        "data_fn": lambda: torch.rand(7, 3, 12, 12),
        "grads": ["All", "Input", "Everything", "Nothing"],
    },
    {
        "name": "SiLU",
        "layer_fn": lambda: torch.nn.SiLU(),
        "data_fn": lambda: torch.rand(7, 3, 12, 12),
        "grads": ["All", "Input", "Everything", "Nothing"],
    },
]


def run_single(model, x, name, dirname):
    y = model(x)
    dot = make_dot(
        y.sum(),
        params=dict(model.named_parameters()),
        show_attrs=True,
        show_saved=True,
    )
    dot.render(filename=name, directory=dirname)


def separate_grad_arguments_wrapper(
    model,
    grad_linear_weights: bool = True,
    grad_linear_bias: bool = True,
    grad_conv_weights: bool = True,
    grad_conv_bias: bool = True,
    grad_norm_weights: bool = True,
    grad_norm_bias: bool = True,
    grad_embed_weights: bool = False,
    **kwargs,
):
    return separate_grad_arguments(
        model,
        grad_linear_weights=grad_linear_weights,
        grad_linear_bias=grad_linear_bias,
        grad_conv_weights=grad_conv_weights,
        grad_conv_bias=grad_conv_bias,
        grad_norm_weights=grad_norm_weights,
        grad_norm_bias=grad_norm_bias,
        grad_embed_weights=grad_embed_weights,
    )


if __name__ == "__main__":
    for layer_to_test in (pbar := tqdm(to_test)):
        pbar.set_description(layer_to_test["name"])
        all_grad_cases = select_cases(layer_to_test["grads"])
        for c_name, c in zip(layer_to_test["grads"], all_grad_cases):
            grad_opts = parse_case(c)
            x = layer_to_test["data_fn"]()
            layer = layer_to_test["layer_fn"]()
            memsave_layer = memsave_torch.nn.convert_to_memory_saving(
                layer, clone_params=True
            )
            leafs, no_leafs = separate_grad_arguments_wrapper(
                layer, **grad_opts
            )  # no weights differentiable

            grad_input = False
            x2 = x.clone()
            grad_input = "grad_input" in grad_opts and grad_opts["grad_input"]

            leafs = ([x] if grad_input else []) + leafs
            no_leafs = ([] if grad_input else [x]) + no_leafs

            for leaf in leafs:
                leaf.requires_grad_(True)
            for no_leaf in no_leafs:
                no_leaf.requires_grad_(False)

            # TODO: add grad weights case
            leafs, no_leafs = separate_grad_arguments_wrapper(
                memsave_layer, **grad_opts
            )  # no weights differentiable

            leafs = ([x2] if grad_input else []) + leafs
            no_leafs = ([] if grad_input else [x2]) + no_leafs

            for leaf in leafs:
                leaf.requires_grad_(True)
            for no_leaf in no_leafs:
                no_leaf.requires_grad_(False)

            dirname = f"torchviz-output/elementary/{layer_to_test['name']}"
            run_single(layer, x, c_name, dirname)
            run_single(memsave_layer, x2, c_name + "_MemSave", dirname)
