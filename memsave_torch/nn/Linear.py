"""Implementation of a memory saving Linear layer.

This is done by not saving the inputs/weights if weight/inputs dont require grad.
"""

import torch
import torch.nn as nn


class MemSaveLinear(nn.Linear):
    """MemSaveLinear."""

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        """Inits a MemSaveLinear layer with the given params.

        Args:
            in_features: in_features
            out_features: out_features
            bias: bias
            device: device
            dtype: dtype
        """
        super().__init__(in_features, out_features, bias, device, dtype)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input to the network [B, F_in]

        Returns:
            torch.Tensor: Output [B, F_out]
        """
        return linearMemSave(x, self.weight, self.bias)

    @classmethod
    def from_nn_Linear(cls, linear: nn.Linear):
        """Converts a nn.Linear layer to MemSaveLinear.

        Args:
            linear : The nn.Linear layer

        Returns:
            obj: The MemSaveLinear object
        """
        obj = cls(
            linear.in_features,
            linear.out_features,
            True if linear.bias is not None else False,
            device=getattr(linear, "device", None),
            dtype=getattr(linear, "dtype", None),
        )
        obj.weight = linear.weight
        obj.bias = linear.bias
        return obj


class _MemSaveLinear(torch.autograd.Function):
    @staticmethod
    def forward(x, weight, bias):
        return nn.functional.linear(x, weight, bias)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, weight, bias = inputs
        need_grad = []
        if ctx.needs_input_grad[0]:
            need_grad.append(weight)
        if ctx.needs_input_grad[1]:
            need_grad.append(x)
        # bias doesnt need anything for calc

        ctx.save_for_backward(*need_grad)

    @staticmethod
    def backward(ctx, grad_output):
        x = weight = None
        current_idx = 0
        if ctx.needs_input_grad[0]:
            # print('0 needs weight')
            weight = ctx.saved_tensors[current_idx]
            current_idx += 1
        if ctx.needs_input_grad[1]:
            # print('1 needs x')
            x = ctx.saved_tensors[current_idx]
            current_idx += 1

        # print(current_idx)

        grad_x = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_x = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(x)
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_x, grad_weight, grad_bias


def linearMemSave(x, weight, bias=None):
    """Functional form of the memory saving linear.

    Args:
        x: Input to the network [B, F_in]
        weight: weight
        bias: bias

    Returns:
        torch.Tensor: Output of the network [B, F_out]
    """
    return _MemSaveLinear.apply(x, weight, bias)
