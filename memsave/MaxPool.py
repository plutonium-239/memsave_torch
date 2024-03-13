"""Implementation of a memory saving MaxPool2d layer.

MaxPool is not trainable but it still needs to store the input size to output appropriate gradients for it's input.
https://discuss.pytorch.org/t/why-does-pytorchs-max-pooling-layer-store-input-tensors/173955/2
It only needs to store the size but the builtin implementation stores the whole input - this is fine when conv layers
are saving the input anyway, but we want to use it with MemSaveConv2d
"""

import torch
import torch.nn as nn
from typing import Union, Tuple


class MemSaveMaxPool2d(nn.MaxPool2d):
    """MemSaveMaxPool2d."""

    def __init__(
        self,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        return_indices=False,
        ceil_mode=False
    ) -> None:
        """Inits a Conv2d layer with the given params.
        
        Args:
            kernel_size: kernel_size
            stride: stride
            padding: padding
            dilation: dilation
            return_indices: return_indices 
            ceil_mode: ceil_mode
        """
        super().__init__(
            kernel_size,
            stride,
            padding,
            dilation,
            return_indices,
            ceil_mode
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input: Input to the network [B, C_in, H, W]

        Returns:
            torch.Tensor: Output [B, C_out, H_out, W_out]
        """
        return maxpool2dMemSave(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
            self.return_indices
        )

    @classmethod
    def from_nn_MaxPool2d(cls, maxpool2d: nn.MaxPool2d):
        """Converts a nn.MaxPool2d layer to MemSaveMaxPool2d.

        Args:
            maxpool2d : The nn.MaxPool2d layer

        Returns:
            obj: The MemSaveMaxPool2d object
        """
        obj = cls(
            maxpool2d.kernel_size,
            maxpool2d.stride,
            maxpool2d.padding,
            maxpool2d.dilation,
            maxpool2d.ceil_mode,
            maxpool2d.return_indices
        )
        return obj


class _MemSaveMaxPool2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, kernel_size, stride, padding, dilation, ceil_mode):
        # print('setting up context', ctx.needs_input_grad)
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.ceil_mode = ceil_mode
        ctx.x_shape = x.shape
        ctx.device = x.device

        # we need indices for backward anyway
        out, indices = nn.functional.max_pool2d(x, kernel_size, stride, padding, dilation, ceil_mode, return_indices=True)
        # this is the same as calling:
        # out,indices = torch.ops.aten.max_pool2d_with_indices(x, kernel_size, stride, padding, dilation, ceil_mode)
        if ctx.needs_input_grad[0]:
            ctx.indices = indices
            ctx.mark_non_differentiable(indices)
        return out, indices

# TODO: save x.dtype, avgpool
    @staticmethod
    def backward(ctx, grad_output, ignored_grad_indices):
        grad_x = None
        if ctx.needs_input_grad[0]:
            x = torch.zeros(ctx.x_shape, device=ctx.device)

            grad_x = torch.ops.aten.max_pool2d_with_indices_backward(
                grad_output,
                x,
                ctx.kernel_size,
                ctx.stride,
                ctx.padding,
                ctx.dilation,
                ctx.ceil_mode,
                ctx.indices
            )

            # print('grads are ', (grad_x is not None), (grad_weight is not None), (grad_bias is not None))

        return grad_x, None, None, None, None, None

# TODO: MaxPool2d doesn't respect requires_grad of a tensor when called separately??

def maxpool2dMemSave(
    input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Functional form of the memory saving max-pooling.
    
    Args:
        input: input [B, C, H, W]
        kernel_size: kernel_size
        stride: stride
        padding: padding
        dilation: dilation
        ceil_mode: ceil_mode
        return_indices: return_indices
    
    Returns:
        if return_indices:
            out, indx: (Output of the maxpool operation [B, C, H_out, W_out], indices of maxpool)
        else:
            out: Output of the maxpool operation [B, C, H_out, W_out]
    """
    out, indx = _MemSaveMaxPool2d.apply(input, kernel_size, stride, padding, dilation, ceil_mode)
    if return_indices:
        return out, indx
    return out
