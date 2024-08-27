Models
===================

This module defines mappings from strings to models. This was necessary to isolate the model being run in :mod:`memsave_torch.util.estimate` and having a separate torch runtime for every single run. Otherwise, CUDA does not clear memory unless absolutely required, even on calling the :func:`torch.cuda.empty_cache()` function, which makes memory measurements very difficult.

.. automodule:: experiments.util.models
   :members: prefix_in_pairs

.. currentmodule:: experiments.util.models

.. attribute:: conv_model_fns

   All listed models have their MemSave variants accessible as ``'memsave_<Model Name>'``

   .. dict2table:: experiments.util.models.conv_model_fns
      :caption: All conv models defined to be used as strings in :mod:`experiments.paper_demo` script
      :filter-out: memsave_,model
      :headers: "Model Name (:class:`str`)", "Model Function"
      :widths: 4, 8

.. attribute:: hf_transformers_models_map

   All listed models have their MemSave variants accessible as ``'memsave_<Model Name>'``

   .. dict2table:: experiments.util.models.hf_transformers_models_map
      :caption: All transformer models defined to be used as strings in :mod:`experiments.paper_demo` script
      :filter-out: memsave_
      :headers: "Model Name (:class:`str`)", "Model Class", "Model Page"
      :widths: 4, 3, 7
