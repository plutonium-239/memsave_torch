"""Implementation of a memory saving LayerNorm.

This is done by not saving the inputs/weights if weight/inputs dont require grad.
"""

import torch
import torch.nn as nn

from memsave_torch.nn.functional import layer_normMemSave


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
