"""Tests for the whole memsave_torch.nn module (all layers and the convert function)"""

from typing import Callable, Dict, Union

import pytest
import torch
import transformers

import memsave_torch

devices = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda")

torch.manual_seed(239)

cases = [
    {
        "name": "Linear1dims",
        "layer_fn": lambda: torch.nn.Linear(3, 5),
        "data_fn": lambda: torch.rand(7, 3),
    },
    {
        "name": "Linear2dims",
        "layer_fn": lambda: torch.nn.Linear(3, 5),
        "data_fn": lambda: torch.rand(7, 12, 3),  # weight sharing
    },
    {
        "name": "Linear3dims",
        "layer_fn": lambda: torch.nn.Linear(3, 5),
        "data_fn": lambda: torch.rand(7, 12, 12, 3),  # weight sharing
    },
    {
        "name": "Conv2d",
        "layer_fn": lambda: torch.nn.Conv2d(3, 5, 3),
        "data_fn": lambda: torch.rand(7, 3, 12, 12),
    },
    {
        "name": "Conv1d",
        "layer_fn": lambda: torch.nn.Conv1d(3, 5, 3),
        "data_fn": lambda: torch.rand(7, 3, 12),
    },
    {
        "name": "BatchNorm2d",
        "layer_fn": lambda: torch.nn.BatchNorm2d(3),
        "data_fn": lambda: torch.rand(7, 3, 12, 12),
    },
    {
        "name": "LayerNorm",
        "layer_fn": lambda: torch.nn.LayerNorm([3, 12, 12]),
        "data_fn": lambda: torch.rand(7, 3, 12, 12),
    },
    {
    # TODO: add testing for dropout (save and load rng state)
    # {
    #     "name": "Dropout"
    #     "layer_fn": lambda: torch.nn.Dropout(),
    #     "data_fn": lambda: torch.rand(7, 3, 12, 12),
    # },
    {
        "name": "MaxPool2d",
        "layer_fn": lambda: torch.nn.MaxPool2d(3),
        "data_fn": lambda: torch.rand(7, 3, 12, 12),
    },
    {
        "name": "ReLU",
        "layer_fn": lambda: torch.nn.ReLU(),
        "data_fn": lambda: torch.rand(7, 3, 12, 12),
    },
]


@pytest.mark.quick
@pytest.mark.parametrize("case", cases, ids=[case["name"] for case in cases])
@pytest.mark.parametrize("device", devices)
def test_single_layer(
    case: Dict[str, Union[str, Callable[[], Union[torch.Tensor, torch.nn.Module]]]],
    device: str,
):
    """Runs tests for the layer_cls defined by `layer`

    This tests for equality of outputs on forward pass and equality of the gradients
    on backward pass, for all parameters and input as well.

    Args:
        case (Dict[str, Callable[[Union[torch.Tensor, torch.nn.Module]]]]): Case dictionary specifying layer_fn and data_fn
        device (str): device
    """
    x = case["data_fn"]().to(device)
    layer = case["layer_fn"]()
    layer.to(device)
    memsave_layer = memsave_torch.nn.convert_to_memory_saving(layer, clone_params=True)

    x1 = x.clone().detach()
    x1.requires_grad = True
    y1 = layer(x1)
    y1.sum().backward()

    x2 = x.clone().detach()
    x2.requires_grad = True
    y2 = memsave_layer(x2)
    y2.sum().backward()

    if device == "cpu":
        atol = 1e-8  # defaults
        rtol = 1e-5  # defaults
    elif device == "cuda":
        atol = 1e-5
        rtol = 1e-4
    assert torch.allclose(y1, y2, rtol=rtol, atol=atol)
    assert torch.allclose(x1.grad, x2.grad, rtol=rtol, atol=atol)
    for p1, p2 in zip(layer.parameters(), memsave_layer.parameters()):
        assert torch.allclose(p1.grad, p2.grad, rtol=rtol, atol=atol)
