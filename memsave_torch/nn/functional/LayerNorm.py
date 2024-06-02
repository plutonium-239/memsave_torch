"""Implementation of a memory saving LayerNorm.

This is done by not saving the inputs/weights if weight/inputs dont require grad.
"""

import torch


class _MemSaveLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps):
        """torch.native_layer_norm is the same as torch.ops.aten.native_layer_norm

        Also, we need to fuse forward and setup_context here because
        we dont want to make save_mean and save_invstd as outputs but need to save them in ctx
        """
        outputs = torch.native_layer_norm(x, normalized_shape, weight, bias, eps)

        ctx.mean = outputs[1]
        ctx.rstd = outputs[2]
        ctx.eps = eps
        ctx.x_shape = x.shape
        ctx.normalized_shape = normalized_shape
        ctx.device = x.device

        need_grad = []  # save_mean and save_invstd
        if ctx.needs_input_grad[0]:
            need_grad.append(weight)
        if any(ctx.needs_input_grad):
            need_grad.append(x)
        # bias doesnt need anything for calc

        ctx.save_for_backward(*need_grad)

        return outputs[0]

    @staticmethod
    def backward(ctx, grad_output):
        x = weight = None
        current_idx = 0
        if ctx.needs_input_grad[0]:
            weight = ctx.saved_tensors[current_idx]
            current_idx += 1
            x = ctx.saved_tensors[current_idx]
        if ctx.needs_input_grad[3]:
            x = ctx.saved_tensors[current_idx]

        if x is None:
            x = torch.zeros(ctx.x_shape, device=ctx.device)
        if weight is None:
            weight = torch.zeros(ctx.normalized_shape, device=ctx.device)
        bias = torch.zeros(ctx.normalized_shape, device=ctx.device)

        grad_x, grad_weight, grad_bias = torch.ops.aten.native_layer_norm_backward(
            grad_output,
            x,
            ctx.normalized_shape,
            ctx.mean,
            ctx.rstd,
            weight,
            bias,
            [ctx.needs_input_grad[0], ctx.needs_input_grad[2], ctx.needs_input_grad[3]],
        )

        return grad_x, None, grad_weight, grad_bias, None


def layer_normMemSave(
    input, normalized_shape, weight=None, bias=None, eps=1e-05
) -> torch.Tensor:
    """Functional form of the memory saving layer_norm.

    Args:
        input: Input to the network [B, C, H, W]
        normalized_shape: normalized_shape
        weight: weight
        bias: bias
        eps: eps
    Returns:
        torch.Tensor: Output of the network [B, C, H, W]
    """
    return _MemSaveLayerNorm.apply(input, normalized_shape, weight, bias, eps)


class _MemSaveRMSLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        rms_norm_inv = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

        need_grad = []
        if ctx.needs_input_grad[0]:
            need_grad.append(weight)
        if any(ctx.needs_input_grad):
            need_grad.append(x)
        # ctx.rms_norm_inv = rms_norm_inv
        ctx.eps = eps

        ctx.save_for_backward(*need_grad)

        # import ipdb; ipdb.set_trace()

        return weight * x * rms_norm_inv

    @staticmethod
    def backward(ctx, grad_output):
        x = weight = None

        grad_x, grad_weight = None, None

        current_idx = 0
        rms_norm_inv = None
        if ctx.needs_input_grad[0]:
            weight = ctx.saved_tensors[current_idx]
            current_idx += 1
            x = ctx.saved_tensors[current_idx]
            x_sq_sum = x.pow(2).sum(-1, keepdims=True)
            n = x.shape[-1]
            rms_norm_inv = torch.rsqrt(x_sq_sum / n + ctx.eps)
            grad_x = (
                grad_output
                * weight
                * (1 / n)
                * rms_norm_inv.pow(3)
                * (x_sq_sum - x * (weight * x).sum(dim=-1, keepdims=True))
            )
        if ctx.needs_input_grad[1]:
            x = ctx.saved_tensors[current_idx]
            if rms_norm_inv is None:
                rms_norm_inv = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + ctx.eps)
            current_idx += 1
            grad_weight = grad_output * x * rms_norm_inv

        return grad_x, grad_weight, None


def rms_normMemSave(input, weight, eps=1e-05) -> torch.Tensor:
    """Functional form of the memory saving layer_norm.

    Args:
        input: Input to the network [B, C, H, W]
        normalized_shape: normalized_shape
        weight: weight
        bias: bias
        eps: eps
    Returns:
        torch.Tensor: Output of the network [B, C, H, W]
    """
    return _MemSaveRMSLayerNorm.apply(input, weight, eps)
