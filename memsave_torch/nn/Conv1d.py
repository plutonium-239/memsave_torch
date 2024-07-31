"""Implementation of a memory saving Conv1d layer.

This is done by not saving the inputs/weights if weight/inputs dont require grad.
"""

import torch
import torch.nn as nn

from memsave_torch.nn.functional import convMemSave


class MemSaveConv1d(nn.Conv1d):
    """MemSaveConv1d."""

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
        """Inits a Conv1d layer with the given params.

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
    def from_nn_Conv1d(cls, conv1d: nn.Conv1d):
        """Converts a nn.Conv1d layer to MemSaveConv1d.

        Args:
            conv1d : The nn.Conv1d layer

        Returns:
            MemSaveConv1d: The MemSaveConv1d object
        """
        obj = cls(
            conv1d.in_channels,
            conv1d.out_channels,
            conv1d.kernel_size,
            conv1d.stride,
            conv1d.padding,
            conv1d.dilation,
            conv1d.groups,
            True if conv1d.bias is not None else False,
            conv1d.padding_mode,
            device=getattr(conv1d, "device", None),
            dtype=getattr(conv1d, "dtype", None),
        )
        obj.weight = conv1d.weight
        obj.bias = conv1d.bias
        return obj
