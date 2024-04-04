Models
===================

This module defines mappings from strings to models. This was necessary to isolate the model being run in :mod:`memsave_torch.util.estimate` and having a separate torch runtime for every single run. Otherwise, CUDA does not clear memory unless absolutely required, even on calling the :func:`torch.cuda.empty_cache()` function, which makes memory measurements very difficult.

.. automodule:: memsave_torch.util.models
   :members: prefix_in_pairs

.. autodata:: memsave_torch.util.models.conv_model_fns

