"""Implementation of a memory saving LayerNorm.

This is done by not saving the inputs/weights if weight/inputs dont require grad.
"""

import torch
import torch.nn as nn

from memsave_torch.nn.functional import layer_normMemSave, rms_normMemSave


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


class RMSLayerNorm(nn.Module):
    """RMS Layer Norm (https://arxiv.org/abs/1910.07467)

    not in torch yet (soon: https://github.com/pytorch/pytorch/issues/72643)
    but used by many LLMs, implementation from https://github.com/huggingface/transformers/blob/v4.39.3/src/transformers/models/t5/modeling_t5.py
    """

    def __init__(self, hidden_size, eps=1e-6):
        """Construct a layernorm module in the T5 style. No bias and no subtraction of mean."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        """Forward pass

        T5 uses a layer_norm which only scales and doesn't shift, thus varience is calculated
        w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        half-precision inputs is done in fp32

        Args:
            hidden_states: Input to the network

        Returns:
            torch.Tensor: Output
        """
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        # import ipdb; ipdb.set_trace()

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class MemSaveRMSLayerNorm(RMSLayerNorm):
    """MemSaveLayerNorm."""

    def __init__(
        self,
        hidden_size,
        eps=1e-06,
        # device=None,
        # dtype=None,
    ):
        """Inits a RMSLayerNorm layer with the given params

        Args:
            hidden_size: hidden_size
            eps: eps
            device: device
            dtype: dtype
        """
        super().__init__(hidden_size, eps)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input to the network [B, C, H, W]

        Returns:
            torch.Tensor: Output [B, C, H, W]
        """
        return rms_normMemSave(x, self.weight, self.eps)

    @classmethod
    def from_existing(cls, ln):
        """Converts a RMS Layer Norm layer (multiple classes) to MemSaveRMSLayerNorm.

        Args:
            ln : The input layer (can be of many types, check `nn.__init__` file)

        Returns:
            obj: The MemSaveRMSLayerNorm object
        """
        obj = cls(
            ln.weight.shape,
            ln.eps,
            # device=getattr(ln, "device", None),
            # dtype=getattr(ln, "dtype", None),
        )
        obj.weight = ln.weight
        return obj
