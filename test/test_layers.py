"""Tests for the whole memsave_torch.nn module (all layers and the convert function)"""

from typing import Type

import pytest
import torch

import memsave_torch

devices = ['cpu']
if torch.cuda.is_available():
    devices.append('cuda')

@pytest.mark.parametrize('device', devices)
@pytest.mark.quick
def test_all(device: str):
    """Runs the `single_layer` test for all layers in `layers_to_test`
    
    Args:
        device (str): device
    """
    layers_to_test = [
        torch.nn.Linear,
        torch.nn.Conv2d,
        torch.nn.Conv1d,
        torch.nn.BatchNorm2d,
        torch.nn.LayerNorm,
        torch.nn.MaxPool2d,
        torch.nn.ReLU,
    ]
    x = torch.rand(7, 3, 12, 12, device=device)
    for layer in layers_to_test:
        assert single_layer(layer, x, device)


def single_layer(layer_cls: Type[torch.nn.Module], x: torch.Tensor, device: str) -> bool:
    """Runs tests for the layer defined by `layer_cls`
    
    This tests for equality of outputs on forward pass and equality of the gradients
    on backward pass, for all parameters and input as well.
    
    Args:
        layer_cls (Type[torch.nn.Module]): torch.nn layer to test it's memsave counterpart
        x (torch.Tensor): Input tensor (B, C, H, W); will be reshaped properly based on layer
        device (str): device
    
    Returns:
        bool: Description
    """
    if layer_cls in [torch.nn.Conv2d]:
        layer = layer_cls(3, 5, 3)
    elif layer_cls in [torch.nn.Conv1d]:
        layer = layer_cls(3, 5, 3)
        x = x[:, :, :, 0]
    elif layer_cls in [torch.nn.BatchNorm2d, torch.nn.MaxPool2d]:
        layer = layer_cls(3)
    elif layer_cls == torch.nn.LayerNorm:
        layer = layer_cls((3, 12, 12))
    elif layer_cls == torch.nn.ReLU:
        layer = layer_cls()
    elif layer_cls == torch.nn.Linear:
        layer = layer_cls(3, 5)
        x = x[:, :, 0, 0]
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

    if device == 'cpu':
        tol = 0
    elif device == 'cuda':
        tol = 1e-5
    assert torch.allclose(y1, y2, atol=tol)
    assert torch.allclose(x1.grad, x2.grad, atol=tol)
    for p1, p2 in zip(layer.parameters(), memsave_layer.parameters()):
        assert torch.allclose(p1.grad, p2.grad, atol=tol)

    return True
