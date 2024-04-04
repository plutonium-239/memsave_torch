nn
===

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

   .. autoclass:: MemSaveConv2d
      :members:

   .. autoclass:: MemSaveLinear
      :members:
   
   Activations and Pooling Layers
   --------------------------------
   .. autoclass:: MemSaveReLU
      :members:

   .. autoclass:: MemSaveMaxPool2d
      :members:

   Normalization Layers
   --------------------------
   .. autoclass:: MemSaveBatchNorm2d
      :members:

   .. autoclass:: MemSaveLayerNorm
      :members:
