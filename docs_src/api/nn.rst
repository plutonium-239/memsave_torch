nn
===

This module tries to mirror the :mod:`torch.nn` module, offering all available layers to be readily replaced by calling the :func:`convert_to_memory_saving()` function.
This module contains the following members:

* :func:`convert_to_memory_saving <memsave_torch.nn.convert_to_memory_saving>`
* :class:`MemSaveConv2d <memsave_torch.nn.MemSaveConv2d>`
* :class:`MemSaveLinear <memsave_torch.nn.MemSaveLinear>`
* :class:`MemSaveReLU <memsave_torch.nn.MemSaveReLU>`
* :class:`MemSaveMaxPool2d <memsave_torch.nn.MemSaveMaxPool2d>`
* :class:`MemSaveBatchNorm2d <memsave_torch.nn.MemSaveBatchNorm2d>`
* :class:`MemSaveLayerNorm <memsave_torch.nn.MemSaveLayerNorm>`


.. module:: memsave_torch.nn
   
.. autofunction:: convert_to_memory_saving

Learnable Layers
------------------

.. autosummary::
   :toctree: nn
   :template: torch_module_extension.rst
   :nosignatures:

   MemSaveConv2d
   MemSaveLinear


Activations and Pooling Layers
--------------------------------

.. autosummary::
   :toctree: nn
   :template: torch_module_extension.rst
   :nosignatures:

   MemSaveReLU
   MemSaveMaxPool2d

Normalization Layers
--------------------------

.. autosummary::
   :toctree: nn
   :template: torch_module_extension.rst
   :nosignatures:

   MemSaveBatchNorm2d
   MemSaveLayerNorm
