API
===

.. toctree::
   :maxdepth: 1
   :caption: Available Modules

   nn
   util/index

.. automodule:: memsave_torch
   
   The ``memsave_torch`` package consists of two main modules :mod:`nn <memsave_torch.nn>` and :mod:`util <memsave_torch.util>` 

   :mod:`memsave_torch.nn`
   ========================
   This module tries to mirror the ``torch.nn`` module, offering the following layers to be readily replaced by calling the :func:`convert_to_memory_saving()` function:

