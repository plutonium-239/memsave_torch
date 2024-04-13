Installation / Quickstart
==========================

.. code:: bash

	pip install git+https://github.com/plutonium-239/memsave_torch


Replace all (valid) layers with ``MemSave`` layers
--------------------------------------------------

The :func:`convert_to_memory_saving` function from the :mod:`memsave_torch.nn` module is a handy tool to replace all layers of a model that is passed to it with their memory saving counterparts.

.. code-block:: python
	:emphasize-lines: 4,5

	import torch
	my_torch_model: torch.nn.Module

	from memsave_torch.nn import convert_to_memory_saving
	memsave_torch_model = convert_to_memory_saving(my_torch_model)

