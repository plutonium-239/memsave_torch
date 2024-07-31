"""Implementation of a memory saving Conv3d layer.

This is done by not saving the inputs/weights if weight/inputs dont require grad.
"""

import torch
import torch.nn as nn

from memsave_torch.nn.functional import convMemSave


class MemSaveConv3d(nn.Conv3d):
    """MemSaveConv3d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        """Inits a Conv3d layer with the given params.

        Args:
            in_channels: in_channels
            out_channels: out_channels
            kernel_size: kernel_size
            stride: stride
            padding: padding
            dilation: dilation
            groups: groups
            bias: bias
            padding_mode: padding_mode
            device: device
            dtype: dtype
        """
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )

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
    def from_nn_Conv3d(cls, conv3d: nn.Conv3d):
        """Converts a nn.Conv3d layer to MemSaveConv3d.

        Args:
            conv3d (nn.Conv3d): The nn.Conv3d layer

        Returns:
            MemSaveConv3d: The MemSaveConv3d object
        """
        obj = cls(
            conv3d.in_channels,
            conv3d.out_channels,
            conv3d.kernel_size,
            conv3d.stride,
            conv3d.padding,
            conv3d.dilation,
            conv3d.groups,
            True if conv3d.bias is not None else False,
            conv3d.padding_mode,
            device=getattr(conv3d, "device", None),
            dtype=getattr(conv3d, "dtype", None),
        )
        obj.weight = conv3d.weight
        obj.bias = conv3d.bias
        return obj
