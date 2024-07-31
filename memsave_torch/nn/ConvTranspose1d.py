"""Implementation of a memory saving 1d transpose convolution layer."""

import torch
import torch.nn as nn

from memsave_torch.nn.functional import convMemSave


class MemSaveConvTranspose1d(nn.ConvTranspose1d):
    """Differentiability-agnostic 1d transpose convolution layer."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input (torch.Tensor): Input to the network [B, C_in, W]

        Returns:
            torch.Tensor: Output [B, C_out, W_out]
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
    def from_nn_ConvTranspose1d(cls, convT1d: nn.ConvTranspose1d):
        """Converts a nn.ConvTranspose1d layer to MemSaveConvTranspose1d.

        Args:
            convT1d (nn.ConvTranspose1d): The nn.ConvTranspose1d layer

        Returns:
            MemSaveConvTranspose1d: The MemSaveConvTranspose1d object
        """
        obj = cls(
            convT1d.in_channels,
            convT1d.out_channels,
            convT1d.kernel_size,
            convT1d.stride,
            convT1d.padding,
            convT1d.output_padding,
            convT1d.groups,
            True if convT1d.bias is not None else False,
            convT1d.dilation,
            convT1d.padding_mode,
            device=getattr(convT1d, "device", None),
            dtype=getattr(convT1d, "dtype", None),
        )
        obj.weight = convT1d.weight
        obj.bias = convT1d.bias
        return obj
