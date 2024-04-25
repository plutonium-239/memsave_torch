"""Implementation of a memory saving Dropout (sort of).

This is done by not saving the whole input/output `float32` tensor and instead just saving the `bool` mask (8bit).
"""

import torch


# TODO: inplace
class _MemSaveDropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, train):
        out, mask = torch.ops.aten.native_dropout(x, p, train)
        if ctx.needs_input_grad[0]:
            ctx.p = p
            ctx.mask = mask
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None

        if ctx.needs_input_grad[0]:
            grad_x = torch.ops.aten.native_dropout_backward(
                grad_output, ctx.mask, scale=1 / (1 - ctx.p)
            )

        return grad_x


def dropoutMemSave(x, p, training):
    """Functional form of the memory saving dropout.

    Args:
        x: Input to the network
        p: Probability of elements being zeroed
        training: Whether the layer is in training mode (no dropout applied in eval)

    Returns:
        torch.Tensor: Output of the network
    """
    return _MemSaveDropout.apply(x, p, training)
