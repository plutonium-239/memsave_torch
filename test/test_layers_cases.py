"""Cases for test_layers.py"""

from dataclasses import dataclass, field
from typing import Callable, List

import torch


@dataclass
class Case:
    """DataClass for cases

    name: Name of the case to display in pytest (also used as ids)
    layer_fn: Function that returns the layer to test
    data_fn: Function that returns a batch of data for the layer
    input_grads: List of possible values for input gradients (not really used since input is independent; can always have/not have grad)
    wt_grads: List of possible values for weight gradients (e.g. ReLU/MaxPool can't have weight grads)
    """

    name: str
    layer_fn: Callable[[], torch.nn.Module]
    data_fn: Callable[[], torch.Tensor]
    input_grads: List[bool] = field(default_factory=lambda: [False, True])
    wt_grads: List[bool] = field(default_factory=lambda: [False, True])


cases = [
    Case(
        name="Linear1dims",
        layer_fn=lambda: torch.nn.Linear(3, 5),
        data_fn=lambda: torch.rand(7, 3),
    ),
    Case(
        name="Linear2dims",
        layer_fn=lambda: torch.nn.Linear(3, 5),
        data_fn=lambda: torch.rand(7, 12, 3),  # weight sharing
    ),
    Case(
        name="Linear3dims",
        layer_fn=lambda: torch.nn.Linear(3, 5),
        data_fn=lambda: torch.rand(7, 12, 12, 3),  # weight sharing
    ),
    Case(
        name="Conv1d",
        layer_fn=lambda: torch.nn.Conv1d(3, 5, 3),
        data_fn=lambda: torch.rand(7, 3, 12),
    ),
    Case(
        name="Conv2d",
        layer_fn=lambda: torch.nn.Conv2d(3, 5, 3),
        data_fn=lambda: torch.rand(7, 3, 12, 12),
    ),
    Case(
        name="Conv3d",
        layer_fn=lambda: torch.nn.Conv3d(3, 5, 3),
        data_fn=lambda: torch.rand(7, 3, 12, 12, 12),
    ),
    Case(
        name="ConvTranspose1d",
        layer_fn=lambda: torch.nn.ConvTranspose1d(3, 5, 3),
        data_fn=lambda: torch.rand(7, 3, 12),
    ),
    Case(
        name="ConvTranspose2d",
        layer_fn=lambda: torch.nn.ConvTranspose2d(3, 5, 3),
        data_fn=lambda: torch.rand(7, 3, 12, 12),
    ),
    Case(
        name="ConvTranspose3d",
        layer_fn=lambda: torch.nn.ConvTranspose3d(3, 5, 3),
        data_fn=lambda: torch.rand(7, 3, 12, 12, 12),
    ),
    Case(
        name="BatchNorm2d",
        layer_fn=lambda: torch.nn.BatchNorm2d(3),
        data_fn=lambda: torch.rand(7, 3, 12, 12),
    ),
    # TODO: add testing for dropout (save and load rng state)
    # Case(
    #     name = "Dropout"
    #     layer_fn = lambda: torch.nn.Dropout(),
    #     data_fn = lambda: torch.rand(7, 3, 12, 12),
    # ),
    Case(
        name="MaxPool2d",
        layer_fn=lambda: torch.nn.MaxPool2d(3),
        data_fn=lambda: torch.rand(7, 3, 12, 12),
        wt_grads=[False],
    ),
    Case(
        name="ReLU",
        layer_fn=lambda: torch.nn.ReLU(),
        data_fn=lambda: torch.rand(7, 3, 12, 12),
        wt_grads=[False],
    ),
]
