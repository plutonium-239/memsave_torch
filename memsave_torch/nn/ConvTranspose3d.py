"""Implementation of a memory saving 1d transpose convolution layer."""

import torch
import torch.nn as nn

from memsave_torch.nn.functional import convMemSave


class MemSaveConvTranspose3d(nn.ConvTranspose3d):
    """Differentiability-agnostic 3d transpose convolution layer."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input (torch.Tensor): Input to the network [B, C_in, D, H, W]

        Returns:
            torch.Tensor: Output [B, C_out, D_out, H_out, W_out]
        """
        return convMemSave(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.transposed,
            self.output_padding,
        )

    @classmethod
    def from_nn_ConvTranspose3d(cls, convT3d: nn.ConvTranspose3d):
        """Converts a nn.ConvTranspose3d layer to MemSaveConvTranspose3d.

        Args:
            convT3d (nn.ConvTranspose3d): The nn.ConvTranspose3d layer

        Returns:
            MemSaveConvTranspose3d: The MemSaveConvTranspose3d object
        """
        obj = cls(
            convT3d.in_channels,
            convT3d.out_channels,
            convT3d.kernel_size,
            convT3d.stride,
            convT3d.padding,
            convT3d.output_padding,
            convT3d.groups,
            True if convT3d.bias is not None else False,
            convT3d.dilation,
            convT3d.padding_mode,
            device=getattr(convT3d, "device", None),
            dtype=getattr(convT3d, "dtype", None),
        )
        obj.weight = convT3d.weight
        obj.bias = convT3d.bias
        return obj
