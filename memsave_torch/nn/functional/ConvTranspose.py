"""Implementation of memory saving transpose convolution layers.

This is done by not saving the inputs/weights if weight/inputs dont require grad.
"""

import torch


class _MemSaveConvTranspose(torch.autograd.Function):
    @staticmethod
    def forward(x, weight, bias, stride, padding, output_padding, dilation, groups):
        return torch.ops.aten.convolution(
            x,
            weight,
            bias,
            stride,
            padding,
            dilation,
            True,
            output_padding,
            groups,
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, weight, bias, stride, padding, output_padding, dilation, groups = inputs
        need_grad = []
        if ctx.needs_input_grad[0]:
            need_grad.append(weight)
        if ctx.needs_input_grad[1]:
            need_grad.append(x)
        # bias doesnt need anything for calc
        ctx.bias_exists = bias is not None
        ctx.stride = stride
        ctx.padding = padding
        ctx.output_padding = output_padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.x_shape = x.shape
        ctx.weight_shape = weight.shape
        ctx.device = x.device

        ctx.save_for_backward(*need_grad)

    @staticmethod
    def backward(ctx, grad_output):
        x = weight = None

        current_idx = 0
        if ctx.needs_input_grad[0]:
            weight = ctx.saved_tensors[current_idx]
            current_idx += 1
        if ctx.needs_input_grad[1]:
            x = ctx.saved_tensors[current_idx]
            current_idx += 1

        if x is None:
            x = torch.zeros(ctx.x_shape, device=ctx.device)
        if weight is None:
            weight = torch.zeros(ctx.weight_shape, device=ctx.device)

        grad_x, grad_weight, grad_bias = torch.ops.aten.convolution_backward(
            grad_output,
            x,
            weight,
            [weight.shape[0]] if ctx.bias_exists else None,
            ctx.stride,
            ctx.padding,
            ctx.dilation,
            True,
            ctx.output_padding,
            ctx.groups,
            ctx.needs_input_grad[:3],
        )

        return grad_x, grad_weight, grad_bias, None, None, None, None, None, None


def conv_transpose1dMemSave(
    input, weight, bias, stride, padding, output_padding, dilation, groups
) -> torch.Tensor:
    """Functional form of the memory saving transpose convolution.

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
    return _MemSaveConvTranspose.apply(
        input, weight, bias, stride, padding, output_padding, dilation, groups
    )


def conv_transpose2dMemSave(
    input, weight, bias, stride, padding, output_padding, dilation, groups
) -> torch.Tensor:
    """Functional form of the memory saving transpose convolution.

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
    return _MemSaveConvTranspose.apply(
        input, weight, bias, stride, padding, output_padding, dilation, groups
    )


def conv_transpose3dMemSave(
    input, weight, bias, stride, padding, output_padding, dilation, groups
) -> torch.Tensor:
    """Functional form of the memory saving transpose convolution.

    Args:
        input: input [B, C_in, D, H, W]
        weight: weight
        bias: bias
        stride: stride
        padding: padding
        dilation: dilation
        groups: groups

    Returns:
        torch.Tensor: Output of the conv operation [B, C_out, D_out, H_out, W_out]
    """
    return _MemSaveConvTranspose.apply(
        input, weight, bias, stride, padding, output_padding, dilation, groups
    )
