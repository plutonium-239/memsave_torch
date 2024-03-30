"""Implementation of a memory saving BatchNorm2d.

This is done by not saving the inputs/weights if weight/inputs dont require grad.
"""

import torch
import torch.nn as nn


class MemSaveBatchNorm2d(nn.BatchNorm2d):
    """MemSaveBatchNorm2d."""

    def __init__(
        self,
        num_features,
        eps=1e-05,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        device=None,
        dtype=None,
    ):
        """Inits a BatchNorm2d layer with the given params

        Args:
            num_features: num_features
            eps: eps
            momentum: momentum
            affine: affine
            track_running_stats: track_running_stats
            device: device
            dtype: dtype
        """
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, device, dtype
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input to the network [B, C, H, W]

        Returns:
            torch.Tensor: Output [B, C, H, W]
        """
        return batch_normMemSave(
            x,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training,
            self.momentum,
            self.eps,
        )

    @classmethod
    def from_nn_BatchNorm2d(cls, bn2d: nn.BatchNorm2d):
        """Converts a nn.BatchNorm2d layer to MemSaveBatchNorm2d.

        Args:
            bn2d : The nn.BatchNorm2d layer

        Returns:
            obj: The MemSaveBatchNorm2d object
        """
        obj = cls(
            bn2d.num_features,
            bn2d.eps,
            bn2d.momentum,
            bn2d.affine,
            bn2d.track_running_stats,
            device=getattr(bn2d, "device", None),
            dtype=getattr(bn2d, "dtype", None),
        )
        obj.weight = bn2d.weight
        obj.bias = bn2d.bias
        obj.running_mean = bn2d.running_mean
        obj.running_var = bn2d.running_var
        return obj


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

        if weight is not None:
            x = torch.zeros(ctx.x_shape, device=weight.device)
        if x is not None:
            weight = torch.zeros(ctx.weight_shape, device=x.device)

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
):
    """Functional form of the memory saving batch_norm.

    Args:
        input (TYPE): Input to the network [B, C, H, W]
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
