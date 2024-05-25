"""Implementation of a memory saving Dropout (sort of).

This is done by not saving the whole input/output `float32` tensor and instead just saving the `bool` mask (8bit).
"""

import torch


# TODO: inplace
class _MemSaveDropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, train):
        rng = torch.get_rng_state()
        # dont need mask here, so dont call torch.ops, torch.dropout is faster
        out = torch.dropout(x, p, train) 
        if ctx.needs_input_grad[0]:
            ctx.p = p
            ctx.train = train
            ctx.rng = rng
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None

        if ctx.needs_input_grad[0]:
            orig_rng = torch.get_rng_state()
            torch.set_rng_state(ctx.rng)
            mask = torch.empty_like(grad_output)
            mask = mask.bernoulli_(0.5).bool()
            torch.set_rng_state(orig_rng)
            grad_x = grad_output*mask/(1-ctx.p)
            # grad_x = torch.ops.aten.native_dropout_backward(
            #     grad_output, mask, scale=1 / (1 - ctx.p)
            # )

        return grad_x, None, None


def dropoutMemSave(x, p, training) -> torch.Tensor:
    """Functional form of the memory saving dropout.

    Args:
        x: Input to the network
        p: Probability of elements being zeroed
        training: Whether the layer is in training mode (no dropout applied in eval)

    Returns:
        torch.Tensor: Output of the network
    """
    return _MemSaveDropout.apply(x, p, training)
