"""Implementation of a memory saving Conv2d layer.

This is done by not saving the inputs/weights if weight/inputs dont require grad.
"""

import torch
import torch.nn as nn

from memsave_torch.nn.functional import convMemSave


class MemSaveConv2d(nn.Conv2d):
    """MemSaveConv2d."""

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
        """Inits a Conv2d layer with the given params.

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
            input: Input to the network [B, C_in, H, W]

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
    def from_nn_Conv2d(cls, conv2d: nn.Conv2d):
        """Converts a nn.Conv2d layer to MemSaveConv2d.

        Args:
            conv2d (nn.Conv2d): The nn.Conv2d layer

        Returns:
            MemSaveConv2d: The MemSaveConv2d object
        """
        obj = cls(
            conv2d.in_channels,
            conv2d.out_channels,
            conv2d.kernel_size,
            conv2d.stride,
            conv2d.padding,
            conv2d.dilation,
            conv2d.groups,
            True if conv2d.bias is not None else False,
            conv2d.padding_mode,
            device=getattr(conv2d, "device", None),
            dtype=getattr(conv2d, "dtype", None),
        )
        obj.weight = conv2d.weight
        obj.bias = conv2d.bias
        return obj
