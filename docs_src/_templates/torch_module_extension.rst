{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
	:members:

.. hint::

	The usage is the same as :class:`torch.nn.{{ objname|replace('MemSave', '') }}`
	
	For usage examples, please refer to the linked ``torch`` documentation
