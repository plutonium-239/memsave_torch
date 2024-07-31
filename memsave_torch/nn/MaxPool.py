"""Implementation of a memory saving MaxPool2d layer.

MaxPool is not trainable but it still needs to store the input size to output appropriate gradients for it's input.
https://discuss.pytorch.org/t/why-does-pytorchs-max-pooling-layer-store-input-tensors/173955/2
It only needs to store the size but the builtin implementation stores the whole input - this is fine when conv layers
are saving the input anyway, but we want to use it with MemSaveConv2d
"""

import torch
import torch.nn as nn

from memsave_torch.nn.functional import maxpool2dMemSave


class MemSaveMaxPool2d(nn.MaxPool2d):
    """MemSaveMaxPool2d."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input: Input to the network [B, C_in, H, W]

        Returns:
            torch.Tensor: Output [B, C_out, H_out, W_out]
        """
        return maxpool2dMemSave(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
            self.return_indices,
        )

    @classmethod
    def from_nn_MaxPool2d(cls, maxpool2d: nn.MaxPool2d):
        """Converts a nn.MaxPool2d layer to MemSaveMaxPool2d.

        Args:
            maxpool2d : The nn.MaxPool2d layer

        Returns:
            obj: The MemSaveMaxPool2d object
        """
        obj = cls(
            maxpool2d.kernel_size,
            maxpool2d.stride,
            maxpool2d.padding,
            maxpool2d.dilation,
            maxpool2d.ceil_mode,
            maxpool2d.return_indices,
        )
        return obj
