"""Tests for the whole memsave_torch.nn module (all layers and the convert function)"""

from typing import List, Type, Union

import pytest
import torch

import memsave_torch

devices = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda")

x = torch.rand(7, 3, 12, 12)
cases = [
    [torch.nn.Linear, [3, 5], x[:, :, 0, 0]],
    [torch.nn.Linear, [3, 5], x[:, :, :, 0].permute(0, 2, 1)],  # weight sharing
    [torch.nn.Linear, [3, 5], x.permute(0, 2, 3, 1)],  # weight sharing
    [torch.nn.Conv2d, [3, 5, 3], x],
    [torch.nn.Conv1d, [3, 5, 3], x[:, :, 0]],
    [torch.nn.BatchNorm2d, [3], x],
    [torch.nn.LayerNorm, [[3, 12, 12]], x],
    [torch.nn.MaxPool2d, [3], x],
    [torch.nn.ReLU, [], x],
]


@pytest.mark.quick
@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("layer_cls,layer_init,x", cases)
def test_single_layer(
    layer_cls: Type[torch.nn.Module],
    layer_init: List[Union[int, List[int]]],
    x: torch.Tensor,
    device: str,
):
    """Runs tests for the layer_cls defined by `layer`

    This tests for equality of outputs on forward pass and equality of the gradients
    on backward pass, for all parameters and input as well.

    Args:
        layer_cls (Type[torch.nn.Module]): torch.nn layer class to test it's memsave counterpart
        layer_init (List[Union[int, List[int]]]): layer initialization parameters
        x (torch.Tensor): Input tensor (B, C, H, W); will be reshaped properly based on layer
        device (str): device
    """
    x = x.to(device)
    layer = layer_cls(*layer_init)
    layer.to(device)
    memsave_layer = memsave_torch.nn.convert_to_memory_saving(layer, clone_params=True)
    # clone_params is neede here because we want to backprop through both layer and memsave_layer

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
        atol = 1e-4
        rtol = 1e-2
    assert torch.allclose(y1, y2, rtol=rtol, atol=atol)
    assert torch.allclose(x1.grad, x2.grad, rtol=rtol, atol=atol)
    for p1, p2 in zip(layer.parameters(), memsave_layer.parameters()):
        assert torch.allclose(p1.grad, p2.grad, rtol=rtol, atol=atol)
