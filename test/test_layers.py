"""Tests for the whole memsave_torch.nn module (all layers and the convert function)"""

import itertools

import pytest
import torch
from test_layers_cases import Case, cases

import memsave_torch

devices = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda")

torch.manual_seed(239)


@pytest.mark.quick
@pytest.mark.parametrize("case", cases, ids=[case.name for case in cases])
@pytest.mark.parametrize("device", devices)
def test_single_layer(
    case: Case,
    device: str,
):
    """Runs tests for the layer_cls defined by `layer`

    This tests for equality of outputs on forward pass and equality of the gradients
    on backward pass, for all parameters and input as well.

    Args:
        case (Case): Case dictionary specifying layer_fn and data_fn
        device (str): device
    """
    x = case.data_fn().to(device)
    for input_grad, wt_grad in itertools.product(case.input_grads, case.wt_grads):
        layer = case.layer_fn()
        layer.to(device)
        memsave_layer = memsave_torch.nn.convert_to_memory_saving(
            layer, clone_params=True
        )

        for p1, p2 in zip(layer.parameters(), memsave_layer.parameters()):
            p1.requires_grad_(wt_grad)
            p2.requires_grad_(wt_grad)

        x1 = x.clone().detach()
        x1.requires_grad = input_grad
        y1 = layer(x1)
        if input_grad or wt_grad:
            y1.sum().backward()

        x2 = x.clone().detach()
        x2.requires_grad = input_grad
        y2 = memsave_layer(x2)
        if input_grad or wt_grad:
            y2.sum().backward()

        if device == "cpu":
            atol = 1e-8  # defaults
            rtol = 1e-5  # defaults
        elif device == "cuda":
            atol = 1e-5
            rtol = 1e-4
        # if "RMS" in case.name or case.name == "T5LayerNorm":
        #     atol = 1e-4
        assert torch.allclose(y1, y2, rtol=rtol, atol=atol)
        if input_grad:
            assert x1.grad is not None
            assert x2.grad is not None
            assert torch.allclose(x1.grad, x2.grad, rtol=rtol, atol=atol)
        if wt_grad:
            for p1, p2 in zip(layer.parameters(), memsave_layer.parameters()):
                assert p1.grad is not None
                assert p2.grad is not None
                assert torch.allclose(p1.grad, p2.grad, rtol=rtol, atol=atol)
