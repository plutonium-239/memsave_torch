"""Implementation of a memory saving 1d transpose convolution layer."""

import torch
import torch.nn as nn
from memsave_torch.nn.functional import conv_transpose2dMemSave


class MemSaveConvTranspose2d(nn.ConvTranspose2d):
    """Differentiability-agnostic 2d transpose convolution layer."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input: Input to the network [B, C_in, H, W]

        Returns:
            torch.Tensor: Output [B, C_out, H_out, W_out]
        """
        return conv_transpose2dMemSave(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
            self.groups,
        )
