"""Implementation of a memory saving Conv2d layer.

This is done by not saving the inputs/weights if weight/inputs dont require grad.
"""

import torch


class _MemSaveConv(torch.autograd.Function):
    @staticmethod
    def forward(x, weight, bias, stride, padding, dilation, groups):
        return torch.ops.aten.convolution(
            x,
            weight,
            bias,
            stride,
            padding,
            dilation,
            False,
            tuple([0] * len(padding)),
            groups,
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, weight, bias, stride, padding, dilation, groups = inputs
        need_grad = []
        if ctx.needs_input_grad[0]:
            need_grad.append(weight)
        if ctx.needs_input_grad[1]:
            need_grad.append(x)
        # bias doesnt need anything for calc
        ctx.bias_exists = bias is not None
        ctx.stride = stride
        ctx.padding = padding
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
        elif ctx.needs_input_grad[1]:
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
            False,
            [0],
            ctx.groups,
            ctx.needs_input_grad[:3],
        )

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
    return _MemSaveConv.apply(input, weight, bias, stride, padding, dilation, groups)


def conv2dMemSave(
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
    return _MemSaveConv.apply(input, weight, bias, stride, padding, dilation, groups)


def conv3dMemSave(
    input, weight, bias, stride, padding, dilation, groups
) -> torch.Tensor:
    """Functional form of the memory saving convolution.

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
    return _MemSaveConv.apply(input, weight, bias, stride, padding, dilation, groups)
