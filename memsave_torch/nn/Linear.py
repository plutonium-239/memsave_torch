"""Implementation of a memory saving Linear layer.

This is done by not saving the inputs/weights if weight/inputs dont require grad.
"""

import sys

import torch.nn as nn

from memsave_torch.nn.functional import linearMemSave

transformers_imported = False
if "transformers" in sys.modules:
    import transformers

    transformers_imported = True


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
        """Converts a nn.Linear/transformers.Conv1D layer to MemSaveLinear.

        Args:
            linear : The nn.Linear/transformers.Conv1D layer

        Returns:
            obj: The MemSaveLinear object
        """
        isTransformersConv1D = False
        if transformers_imported:
            isTransformersConv1D = isinstance(linear, transformers.Conv1D)
        if isTransformersConv1D:
            # it only saves output features in the model (linear.nf); need to take input features from weight anyway
            # weight and bias are still defined
            linear.in_features, linear.out_features = linear.weight.shape
        obj = cls(
            linear.in_features,
            linear.out_features,
            True if linear.bias is not None else False,
            device=getattr(linear, "device", None),
            dtype=getattr(linear, "dtype", None),
        )
        if isTransformersConv1D:
            obj.weight = nn.Parameter(linear.weight.T)
        else:
            obj.weight = linear.weight
        obj.bias = linear.bias
        return obj
