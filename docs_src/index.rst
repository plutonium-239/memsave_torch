.. :layout: landing

.. MemSave PyTorch documentation master file, created by
   sphinx-quickstart on Tue Apr  2 17:30:57 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

``memsave_torch`` 
===================

Lowering PyTorch's Memory Consumption for Selective Differentiation
**********************************************************************

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Index

   Overview <self>
   basic_usage
   api/index

.. image:: /_static/memsave_torch_banner.svg
   :width: 100%


This package offers drop-in implementations of PyTorch :class:`torch.nn.Module` s.
They are as fast as their built-in equivalents, but more memory-efficient whenever you want to compute gradients for a sub-set of parameters (i.e. some have ``requires_grad=False``).


.. raw:: html

   <details>
   <summary>
      Currently it supports the following layers:
   </summary>

.. .. hlist::
..    :columns: 2

* :class:`memsave_torch.nn.MemSaveConv1d`
* :class:`memsave_torch.nn.MemSaveConv2d`
* :class:`memsave_torch.nn.MemSaveConv3d`
* :class:`memsave_torch.nn.MemSaveReLU`
* :class:`memsave_torch.nn.MemSaveMaxPool2d`
* :class:`memsave_torch.nn.MemSaveConvTranspose1d`
* :class:`memsave_torch.nn.MemSaveConvTranspose2d`
* :class:`memsave_torch.nn.MemSaveConvTranspose3d`
* :class:`memsave_torch.nn.MemSaveBatchNorm2d`

.. raw:: html

   </details>

Also, each layer has a ``.from_nn_<layername>(layer)`` function which allows to convert a single layer into its memory-saving equivalent. (e.g. :func:`MemSaveConv2d.from_nn_Conv2d <memsave_torch.nn.MemSaveConv2d.from_nn_Conv2d>`)

Installation
**************

Normal installation:

.. code-block:: bash

   pip install git+https://github.com/plutonium-239/memsave_torch


Install (editable):

.. code-block:: bash

   pip install -e git+https://github.com/plutonium-239/memsave_torch


Usage
*******

Please refer to :doc:`basic_usage`.

.. Basic Example
.. ***************

.. .. code-block:: python
..    :emphasize-lines: 1,9

..    from memsave_torch.nn import convert_to_memory_saving

..    X, y = ...

..    model = Sequential(...)
..    loss_func = MSELoss()

..    # replace all available layers with mem-saving layers
..    model = memsave_torch.nn.convert_to_memory_saving(model)

..    # same as if you were using the original model
..    loss = loss_func(model(X), y)


Further reading
******************

* `Writeup <https://github.com/plutonium-239/memsave_torch/blob/llm/experiments/writeup.md>`_

   This explains the basic ideas around MemSave without diving into too many details.

* `Our paper (WANT@ICML'24) <https://openreview.net/pdf?id=KsUUzxUK7N>`_
   
   It is also available on `arXiv <https://arxiv.org/abs/2404.12406>`_.

How to cite
*************

If this package has benefited you at some point, consider citing

.. code:: bibtex

   @inproceedings{
      bhatia2024lowering,
      title={Lowering PyTorch's Memory Consumption for Selective Differentiation},
      author={Samarth Bhatia and Felix Dangel},
      booktitle={2nd Workshop on Advancing Neural Network Training: Computational Efficiency, Scalability, and Resource Optimization (WANT@ICML 2024)},
      year={2024},
      url={https://openreview.net/forum?id=KsUUzxUK7N}
   }

Contributors
**************

.. contributors:: plutonium-239/memsave_torch
   :avatars:
   :limit: 5
   :order: ASC


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
