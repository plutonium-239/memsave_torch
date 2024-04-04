"""Implementation of a memory saving Conv1d layer.

This is done by not saving the inputs/weights if weight/inputs dont require grad.
"""

import torch
import torch.nn as nn


class MemSaveConv1d(nn.Conv1d):
    """MemSaveConv1d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        """Inits a Conv1d layer with the given params.

        Args:
            in_channels: in_channels
            out_channels: out_channels
            kernel_size: kernel_size
            stride: stride
            padding: padding
            dilation: dilation
            groups: groups
            bias: bias
            padding_mode: padding_mode
            device: device
            dtype: dtype
        """
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input: Input to the network [B, C_in, H, W]

        Returns:
            torch.Tensor: Output [B, C_out, H_out, W_out]
        """
        return conv1dMemSave(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    @classmethod
    def from_nn_Conv1d(cls, conv1d: nn.Conv1d):
        """Converts a nn.Conv1d layer to MemSaveConv1d.

        Args:
            conv1d : The nn.Conv1d layer

        Returns:
            obj: The MemSaveConv1d object
        """
        obj = cls(
            conv1d.in_channels,
            conv1d.out_channels,
            conv1d.kernel_size,
            conv1d.stride,
            conv1d.padding,
            conv1d.dilation,
            conv1d.groups,
            True if conv1d.bias is not None else False,
            conv1d.padding_mode,
            device=getattr(conv1d, "device", None),
            dtype=getattr(conv1d, "dtype", None),
        )
        obj.weight = conv1d.weight
        obj.bias = conv1d.bias
        return obj


class _MemSaveConv1d(torch.autograd.Function):
    @staticmethod
    def forward(x, weight, bias, stride, padding, dilation, groups):
        return nn.functional.conv1d(x, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, weight, bias, stride, padding, dilation, groups = inputs
        # print('setting up context', ctx.needs_input_grad)
        need_grad = []
        if ctx.needs_input_grad[0]:
            # print('weight saved')
            need_grad.append(weight)
        if ctx.needs_input_grad[1]:
            # print('x saved')
            need_grad.append(x)
        # bias doesnt need anything for calc
        ctx.bias_exists = bias is not None
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.x_shape = x.shape
        ctx.weight_shape = weight.shape

        ctx.save_for_backward(*need_grad)

    @staticmethod
    def backward(ctx, grad_output):
        x = weight = None

        current_idx = 0
        if ctx.needs_input_grad[0]:
            # print('0 needs weight')
            weight = ctx.saved_tensors[current_idx]
            current_idx += 1
        elif ctx.needs_input_grad[1]:
            # print('1 needs x')
            x = ctx.saved_tensors[current_idx]
            current_idx += 1

        if weight is not None:
            x = torch.zeros(ctx.x_shape, device=weight.device)
        if x is not None:
            weight = torch.zeros(ctx.weight_shape, device=x.device)

        # print(current_idx)

        grad_x, grad_weight, grad_bias = torch.ops.aten.convolution_backward(
            grad_output,
            x,
            weight,
            weight.shape[0] if ctx.bias_exists else None,
            ctx.stride,
            ctx.padding,
            ctx.dilation,
            False,
            [0],
            ctx.groups,
            ctx.needs_input_grad[:3],
        )

        # print('grads are ', (grad_x is not None), (grad_weight is not None), (grad_bias is not None))

        return grad_x, grad_weight, grad_bias, None, None, None, None, None


def conv1dMemSave(
    input, weight, bias, stride, padding, dilation, groups
) -> torch.Tensor:
    """Functional form of the memory saving convolution.

    Args:
        input: input [B, C_in, H, W]
        weight: weight
        bias: bias
        stride: stride
        padding: padding
        dilation: dilation
        groups: groups

    Returns:
        torch.Tensor: Output of the conv operation [B, C_out, H_out, W_out]
    """
    return _MemSaveConv1d.apply(input, weight, bias, stride, padding, dilation, groups)
