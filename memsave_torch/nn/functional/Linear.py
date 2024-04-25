"""Implementation of a memory saving Linear layer.

This is done by not saving the inputs/weights if weight/inputs dont require grad.
"""

import torch
import torch.nn as nn


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
            grad_x = grad_output @ weight
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.mT @ x
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
