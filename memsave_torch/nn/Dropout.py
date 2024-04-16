"""Implementation of a memory saving Dropout (sort of).

This is done by not saving the whole input/output `float32` tensor and instead just saving the `bool` mask (8bit).
"""

import torch
import torch.nn as nn


class MemSaveDropout(nn.Dropout):
    """MemSaveDropout."""

    def __init__(self, p=0.5):
        """Inits a MemSaveDropout layer with the given params.

        Args:
            p: Probability of elements being zeroed
        """
        super().__init__(p)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input to the network

        Returns:
            torch.Tensor: Output
        """
        return dropoutMemSave(x, self.p, self.training)

    @classmethod
    def from_nn_dropout(cls, dropout: nn.Dropout):
        """Converts a nn.Dropout layer to MemSaveDropout.

        Args:
            dropout : The nn.Dropout layer

        Returns:
            obj: The MemSaveDropout object
        """
        obj = cls(dropout.p)
        return obj


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
