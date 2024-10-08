"""This module tries to replicate `torch.nn.functional`

The main focus are the functionals which can be imported directly from this module.
These can be used in the same way as pytorch functionals.

This also contains the `torch.autograd.Function`s needed which actually change the backward behaviour.
These are not exported but can still be used as: `from memsave_torch.nn.functional.Linear import _MemSaveLinear`
"""

from memsave_torch.nn.functional.BatchNorm import batch_normMemSave  # noqa: F401
from memsave_torch.nn.functional.Conv import convMemSave  # noqa: F401
from memsave_torch.nn.functional.Dropout import dropoutMemSave  # noqa: F401
from memsave_torch.nn.functional.MaxPool import maxpool2dMemSave  # noqa: F401
from memsave_torch.nn.functional.ReLU import reluMemSave  # noqa: F401
