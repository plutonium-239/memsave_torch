"""Implementation of a memory saving LayerNorm.

This is done by not saving the inputs/weights if weight/inputs dont require grad.
"""

import torch
import torch.nn as nn


class MemSaveLayerNorm(nn.LayerNorm):
    """MemSaveLayerNorm."""

    def __init__(
        self,
        normalized_shape,
        eps=1e-05,
        elementwise_affine=True,
        bias=True,
        device=None,
        dtype=None,
    ):
        """Inits a LayerNorm layer with the given params

        Args:
            normalized_shape: normalized_shape
            eps: eps
            elementwise_affine: elementwise_affine
            bias: bias (introduced in torch v2.1)
            device: device
            dtype: dtype
        """
        if torch.__version__.startswith("2.1"):
            super().__init__(
                normalized_shape,
                eps,
                elementwise_affine,
                bias,
                device,
                dtype,  # type: ignore
            )
        else:
            super().__init__(normalized_shape, eps, elementwise_affine, device, dtype)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input to the network [B, C, H, W]

        Returns:
            torch.Tensor: Output [B, C, H, W]
        """
        return layer_normMemSave(
            x,
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps,
        )

    @classmethod
    def from_nn_LayerNorm(cls, ln: nn.LayerNorm):
        """Converts a nn.LayerNorm layer to MemSaveLayerNorm.

        Args:
            ln : The nn.LayerNorm layer

        Returns:
            obj: The MemSaveLayerNorm object
        """
        obj = cls(
            ln.normalized_shape,
            ln.eps,
            ln.elementwise_affine,
            ln.bias is not None,
            device=getattr(ln, "device", None),
            dtype=getattr(ln, "dtype", None),
        )
        obj.weight = ln.weight
        if ln.bias is None:
            torch_version = float(torch.__version__[:3])
            assert torch_version < 2.1, (
                f"Trying to load a model saved in torch>=2.1, but system version is {torch_version}. \n"
                + "This is problematic because torch 2.1 changed how LayerNorm bias works."
            )
        else:
            obj.bias = ln.bias
        return obj


class _MemSaveLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps):
        """torch.native_layer_norm is the same as torch.ops.aten.native_layer_norm

        Also, we need to fuse forward and setup_context here because
        we dont want to make save_mean and save_invstd as outputs but need to save them in ctx
        """
        outputs = torch.native_layer_norm(x, normalized_shape, weight, bias, eps)

        # print('setting up context', ctx.needs_input_grad)
        ctx.mean = outputs[1]
        ctx.rstd = outputs[2]
        ctx.eps = eps
        ctx.x_shape = x.shape
        ctx.normalized_shape = normalized_shape
        ctx.device = x.device

        need_grad = []  # save_mean and save_invstd
        if ctx.needs_input_grad[0]:
            need_grad.append(weight)
        if ctx.needs_input_grad[2]:
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
            weight = torch.zeros(ctx.normalized_shape, device=ctx.device)
        bias = torch.zeros(ctx.normalized_shape, device=ctx.device)

        # print(current_idx)

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


def layer_normMemSave(input, normalized_shape, weight=None, bias=None, eps=1e-05):
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
