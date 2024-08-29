nn
===

This module tries to mirror the :mod:`torch.nn` module, offering all available layers to be readily replaced by calling the :func:`convert_to_memory_saving()` function.
This module contains the following members:

* :func:`convert_to_memory_saving <memsave_torch.nn.convert_to_memory_saving>`
* :class:`MemSaveConv1d <memsave_torch.nn.MemSaveConv1d>`
* :class:`MemSaveConv2d <memsave_torch.nn.MemSaveConv2d>`
* :class:`MemSaveConv3d <memsave_torch.nn.MemSaveConv3d>`
* :class:`MemSaveConvTranspose1d <memsave_torch.nn.MemSaveConvTranspose2d>`
* :class:`MemSaveConvTranspose2d <memsave_torch.nn.MemSaveConvTranspose2d>`
* :class:`MemSaveConvTranspose3d <memsave_torch.nn.MemSaveConvTranspose2d>`
* :class:`MemSaveReLU <memsave_torch.nn.MemSaveReLU>`
* :class:`MemSaveMaxPool2d <memsave_torch.nn.MemSaveMaxPool2d>`
* :class:`MemSaveBatchNorm2d <memsave_torch.nn.MemSaveBatchNorm2d>`

This module also contains the submodule :mod:`memsave_torch.nn.functional`, which tries to mirror the :mod:`torch.nn.functional` module, offering layer operations as functions.

.. toctree::
   :hidden:

   functional/index

.. module:: memsave_torch.nn
   
.. autofunction:: convert_to_memory_saving

Learnable Layers
------------------

.. autosummary::
   :toctree:
   :template: torch_module_extension.rst
   :nosignatures:

   MemSaveConv1d
   MemSaveConv2d
   MemSaveConv3d
   MemSaveConvTranspose1d
   MemSaveConvTranspose2d
   MemSaveConvTranspose3d


Activations and Pooling Layers
--------------------------------

.. autosummary::
   :toctree:
   :template: torch_module_extension.rst
   :nosignatures:

   MemSaveReLU
   MemSaveMaxPool2d

Normalization Layers
--------------------------

.. autosummary::
   :toctree:
   :template: torch_module_extension.rst
   :nosignatures:

   MemSaveBatchNorm2d
