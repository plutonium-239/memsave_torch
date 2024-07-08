"""Implementation of a memory saving Dropout (sort of).

This is done by not saving the whole input/output `float32` tensor and instead just saving the `bool` mask (8bit).
"""

import torch.nn as nn
from memsave_torch.nn.functional import dropoutMemSave


class MemSaveDropout(nn.Dropout):
    """MemSaveDropout."""

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input to the network

        Returns:
            torch.Tensor: Output
        """
        return dropoutMemSave(x, self.p, self.training)

    @classmethod
    def from_nn_dropout(cls, dropout: nn.Dropout):
        """Converts a nn.Dropout layer to MemSaveDropout.

        Args:
            dropout : The nn.Dropout layer

        Returns:
            obj: The MemSaveDropout object
        """
        obj = cls(dropout.p)
        return obj
