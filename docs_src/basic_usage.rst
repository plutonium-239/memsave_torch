Installation / Quickstart
==========================

.. code:: bash

	pip install git+https://github.com/plutonium-239/memsave_torch


Replace all (valid) layers with ``MemSave`` layers
--------------------------------------------------

The :func:`convert_to_memory_saving <memsave_torch.nn.convert_to_memory_saving>` function from the :mod:`memsave_torch.nn` module is a handy tool to replace all layers of a model that is passed to it with their memory saving counterparts.

.. code-block:: python
	:emphasize-lines: 3,8

	import torch
	from torchvision.models import resnet18
	from memsave_torch.nn import convert_to_memory_saving

	x = torch.rand(2, 3, 224, 224)
	rn18 = resnet18()
	
	rn18 = convert_to_memory_saving(rn18)
	
	# Set input to be differentiable and model weights to be non-differentiable
	x.requires_grad = True
	rn18.requires_grad_(False)

	y = rn18(x)
	loss = torch.nn.MSELoss()(y, torch.rand_like(y))
	loss.backward()
	

.. attention::
	You can't use the old model in the same python run after calling :func:`convert_to_memory_saving <memsave_torch.nn.convert_to_memory_saving>` on it, because by default weights are not copied to not cause extra memory consumption. 

	However, if you need to use both models together, pass the ``clone_params = True`` argument to :func:`convert_to_memory_saving <memsave_torch.nn.convert_to_memory_saving>`, this will cause model weights to be copied and not just referenced.