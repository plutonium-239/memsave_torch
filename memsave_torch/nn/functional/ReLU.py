"""Implementation of a memory saving ReLU (sort of).

This is done by not saving the whole input/output `float32` tensor and instead just saving the `bool` mask (8bit).
"""

import torch


# TODO: inplace
class _MemSaveReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        mask = x > 0
        if ctx.needs_input_grad[0]:
            ctx.mask = mask
        x = x * mask
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None

        if ctx.needs_input_grad[0]:
            grad_x = grad_output * ctx.mask

        return grad_x


def reluMemSave(x) -> torch.Tensor:
    """Functional form of the memory saving relu.

    Args:
        x: Input to the network

    Returns:
        torch.Tensor: Output of the network
    """
    return _MemSaveReLU.apply(x)
