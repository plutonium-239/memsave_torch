"""Implementation of a memory saving Linear layer.

This is done by not saving the inputs/weights if weight/inputs dont require grad.
"""

import torch.nn as nn

from memsave_torch.nn.functional import linearMemSave


class MemSaveLinear(nn.Linear):
    """MemSaveLinear."""

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        """Inits a MemSaveLinear layer with the given params.

        Args:
            in_features: in_features
            out_features: out_features
            bias: bias
            device: device
            dtype: dtype
        """
        super().__init__(in_features, out_features, bias, device, dtype)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input to the network [B, F_in]

        Returns:
            torch.Tensor: Output [B, F_out]
        """
        return linearMemSave(x, self.weight, self.bias)

    @classmethod
    def from_nn_Linear(cls, linear: nn.Linear):
        """Converts a nn.Linear layer to MemSaveLinear.

        Args:
            linear : The nn.Linear layer

        Returns:
            obj: The MemSaveLinear object
        """
        obj = cls(
            linear.in_features,
            linear.out_features,
            True if linear.bias is not None else False,
            device=getattr(linear, "device", None),
            dtype=getattr(linear, "dtype", None),
        )
        obj.weight = linear.weight
        obj.bias = linear.bias
        return obj
