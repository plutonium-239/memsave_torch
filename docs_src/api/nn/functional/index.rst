functional
===========

.. module:: memsave_torch.nn.functional

.. For some reason, linking to :mod:`torch.nn.functional` does not point to the correct documentation

This module tries to mirror the :external+torch:doc:`nn.functional` module, offering layer operations as functions, where you need to provide the inputs to the layer, the layer parameters and any other options it might need.

This module contains the following members:

* :func:`batch_normMemSave`
* :func:`convMemSave`
* :func:`dropoutMemSave`
* :func:`linearMemSave`
* :func:`maxpool2dMemSave`
* :func:`reluMemSave`

.. autofunction:: batch_normMemSave
.. autofunction:: convMemSave
.. autofunction:: dropoutMemSave
.. autofunction:: linearMemSave
.. autofunction:: maxpool2dMemSave
.. autofunction:: reluMemSave
