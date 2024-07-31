"""Implementation of a memory saving 1d transpose convolution layer."""

import torch
import torch.nn as nn

from memsave_torch.nn.functional import convMemSave


class MemSaveConvTranspose2d(nn.ConvTranspose2d):
    """Differentiability-agnostic 2d transpose convolution layer."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input (torch.Tensor): Input to the network [B, C_in, H, W]

        Returns:
            torch.Tensor: Output [B, C_out, H_out, W_out]
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
    def from_nn_ConvTranspose2d(cls, convT2d: nn.ConvTranspose2d):
        """Converts a nn.ConvTranspose2d layer to MemSaveConvTranspose2d.

        Args:
            convT2d (nn.ConvTranspose2d): The nn.ConvTranspose2d layer

        Returns:
            MemSaveConvTranspose2d: The MemSaveConvTranspose2d object
        """
        obj = cls(
            convT2d.in_channels,
            convT2d.out_channels,
            convT2d.kernel_size,
            convT2d.stride,
            convT2d.padding,
            convT2d.output_padding,
            convT2d.groups,
            True if convT2d.bias is not None else False,
            convT2d.dilation,
            convT2d.padding_mode,
            device=getattr(convT2d, "device", None),
            dtype=getattr(convT2d, "dtype", None),
        )
        obj.weight = convT2d.weight
        obj.bias = convT2d.bias
        return obj
