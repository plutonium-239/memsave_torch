"""Implementation of a memory saving ReLU (sort of).

This is done by not saving the whole input/output `float32` tensor and instead just saving the `bool` mask (8bit).
"""

import torch.nn as nn

from memsave_torch.nn.functional import reluMemSave


class MemSaveReLU(nn.ReLU):
    """MemSaveReLU."""

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input to the network

        Returns:
            torch.Tensor: Output
        """
        return reluMemSave(x)

    @classmethod
    def from_nn_ReLU(cls, relu: nn.ReLU):
        """Converts a nn.ReLU layer to MemSaveReLU.

        Args:
            relu : The nn.ReLU layer

        Returns:
            obj: The MemSaveReLU object
        """
        return cls()
