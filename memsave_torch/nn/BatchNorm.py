"""Implementation of a memory saving BatchNorm2d.

This is done by not saving the inputs/weights if weight/inputs dont require grad.
"""

import torch.nn as nn

from memsave_torch.nn.functional import batch_normMemSave


class MemSaveBatchNorm2d(nn.BatchNorm2d):
    """MemSaveBatchNorm2d."""

    def __init__(
        self,
        num_features,
        eps=1e-05,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        device=None,
        dtype=None,
    ):
        """Inits a BatchNorm2d layer with the given params

        Args:
            num_features: num_features
            eps: eps
            momentum: momentum
            affine: affine
            track_running_stats: track_running_stats
            device: device
            dtype: dtype
        """
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, device, dtype
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input to the network [B, C, H, W]

        Returns:
            torch.Tensor: Output [B, C, H, W]
        """
        return batch_normMemSave(
            x,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training,
            self.momentum,
            self.eps,
        )

    @classmethod
    def from_nn_BatchNorm2d(cls, bn2d: nn.BatchNorm2d):
        """Converts a nn.BatchNorm2d layer to MemSaveBatchNorm2d.

        Args:
            bn2d : The nn.BatchNorm2d layer

        Returns:
            obj: The MemSaveBatchNorm2d object
        """
        obj = cls(
            bn2d.num_features,
            bn2d.eps,
            bn2d.momentum,
            bn2d.affine,
            bn2d.track_running_stats,
            device=getattr(bn2d, "device", None),
            dtype=getattr(bn2d, "dtype", None),
        )
        obj.weight = bn2d.weight
        obj.bias = bn2d.bias
        obj.running_mean = bn2d.running_mean
        obj.running_var = bn2d.running_var
        obj.training = bn2d.training
        return obj
