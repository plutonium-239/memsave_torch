"""Implementation of a memory saving ReLU (sort of).

This is done by not saving the whole input/output `float32` tensor and instead just saving the `bool` mask (8bit).
"""

import torch
import torch.nn as nn


class MemSaveReLU(nn.ReLU):
    """MemSaveReLU."""

    def __init__(self):
        """Inits a MemSaveReLU layer with the given params."""
        super().__init__()

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input to the network

        Returns:
            torch.Tensor: Output
        """
        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
        #     with record_function('relu_custom_only'):
        #         res = reluMemSave(x)

        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof2:
        #     with record_function('relu_default_only'):
        #         res = super().forward(x)
        # import ipdb; ipdb.set_trace()
        return reluMemSave(x)

    @classmethod
    def from_nn_ReLU(cls, relu: nn.ReLU):
        """Converts a nn.ReLU layer to MemSaveReLU.

        Args:
            relu : The nn.ReLU layer

        Returns:
            obj: The MemSaveReLU object
        """
        obj = cls()
        return obj


# TODO: inplace
class _MemSaveReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        mask = x > 0
        if ctx.needs_input_grad[0]:
            ctx.mask = mask
        # return nn.functional.relu(x)
        x = x * mask
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None

        if ctx.needs_input_grad[0]:
            # grad_x = grad_output.clone()
            # grad_x[~ctx.mask] = 0
            grad_x = grad_output * ctx.mask

        return grad_x


def reluMemSave(x):
    """Functional form of the memory saving relu.

    Args:
        x: Input to the network

    Returns:
        torch.Tensor: Output of the network
    """
    return _MemSaveReLU.apply(x)
