"""Implementation of a memory saving BatchNorm2d.

This is done by not saving the inputs/weights if weight/inputs dont require grad.
"""

import torch


class _MemSaveBatchNorm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x, running_mean, running_var, weight, bias, training, momentum, eps
    ):
        """torch.native_batch_norm is the same as torch.ops.aten.native_batch_norm

        Not using functional.batch_norm here because we need the `save_mean` and `save_invstd` values
        returned by torch ops in the backward pass (it is the forwarded batch's "stable" mean and invstd)

        Also, we need to fuse forward and setup_context here because
        we dont want to make save_mean and save_invstd as outputs but need to save them in ctx
        """
        outputs = torch.native_batch_norm(
            x, weight, bias, running_mean, running_var, training, momentum, eps
        )

        # print('setting up context', ctx.needs_input_grad)
        ctx.save_mean = outputs[1]
        ctx.save_invstd = outputs[2]
        ctx.running_mean = running_mean
        ctx.running_var = running_var
        ctx.training = training
        ctx.momentum = momentum
        ctx.eps = eps
        ctx.x_shape = x.shape
        ctx.weight_shape = weight.shape
        ctx.device = x.device

        need_grad = []  # save_mean and save_invstd
        if ctx.needs_input_grad[0]:
            need_grad.append(weight)
        if ctx.needs_input_grad[3]:
            need_grad.append(x)
        # bias doesnt need anything for calc

        ctx.save_for_backward(*need_grad)

        return outputs[0]

    @staticmethod
    def backward(ctx, grad_output):
        # print('backward', ctx.needs_input_grad)
        x = weight = None
        current_idx = 0
        if ctx.needs_input_grad[0]:
            weight = ctx.saved_tensors[current_idx]
            current_idx += 1
        if ctx.needs_input_grad[3]:
            x = ctx.saved_tensors[current_idx]
            current_idx += 1

        if x is None:
            x = torch.zeros(ctx.x_shape, device=ctx.device)
        if weight is None:
            weight = torch.zeros(ctx.weight_shape, device=ctx.device)

        # print(current_idx)

        grad_x, grad_weight, grad_bias = torch.ops.aten.native_batch_norm_backward(
            grad_output,
            x,
            weight,
            ctx.running_mean,
            ctx.running_var,
            ctx.save_mean,
            ctx.save_invstd,
            ctx.training,
            ctx.eps,
            [ctx.needs_input_grad[0], ctx.needs_input_grad[3], ctx.needs_input_grad[4]],
        )

        return grad_x, None, None, grad_weight, grad_bias, None, None, None


def batch_normMemSave(
    input,
    running_mean,
    running_var,
    weight=None,
    bias=None,
    training=False,
    momentum=0.1,
    eps=1e-05,
) -> torch.Tensor:
    """Functional form of the memory saving batch_norm.

    Args:
        input: Input to the network [B, C, H, W]
        running_mean: running_mean
        running_var: running_var
        weight: weight
        bias: bias
        training: training
        momentum: momentum
        eps: eps

    Returns:
        torch.Tensor: Output of the network [B, C, H, W]
    """
    return _MemSaveBatchNorm.apply(
        input, running_mean, running_var, weight, bias, training, momentum, eps
    )
