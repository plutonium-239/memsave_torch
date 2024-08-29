# `memsave_torch`: Lowering PyTorch's Memory Consumption for Selective Differentiation

This package offers drop-in implementations of PyTorch `nn.Module`s.
They are as fast as their built-in equivalents, but more memory-efficient whenever you want to compute gradients for a sub-set of parameters (i.e. some have `requires_grad=False`).
You can convert your neural network by calling the [`memsave_torch.nn.convert_to_memory_saving`](https://memsave-torch.readthedocs.io/en/stable/api/nn/index.html#memsave_torch.nn.convert_to_memory_saving) function.

Take a look at the [Basic Example](#basic-example) to see how it works.

Currently it supports the following layers:
- `memsave_torch.nn.MemSaveConv1d`
- `memsave_torch.nn.MemSaveConv2d`
- `memsave_torch.nn.MemSaveConv3d`
- `memsave_torch.nn.MemSaveConvTranspose1d`
- `memsave_torch.nn.MemSaveConvTranspose2d`
- `memsave_torch.nn.MemSaveConvTranspose3d`
- `memsave_torch.nn.MemSaveReLU`
- `memsave_torch.nn.MemSaveBatchNorm2d`
- `memsave_torch.nn.MemSaveMaxPool2d`

Also, each layer has a `.from_nn_<layername>(layer)` function which allows to convert a single `torch.nn` layer into its memory-saving equivalent. (e.g. [`MemSaveConv2d.from_nn_Conv2d`](https://memsave-torch.readthedocs.io/en/stable/api/nn/memsave_torch.nn.MemSaveConv2d.html))

## Installation

Normal installation:
```bash
pip install git+https://github.com/plutonium-239/memsave_torch
```

Install (editable):
```bash
pip install -e .
```

## Basic Example

```python
X, y = ...

model = Sequential(...)
loss_func = MSELoss()

# replace all available layers with mem-saving layers
model = memsave_torch.nn.convert_to_memory_saving(model)

# same as if you were using the original model
loss = loss_func(model(X), y)
```

## Further reading

- [Writeup](https://github.com/plutonium-239/memsave_torch/blob/llm/experiments/writeup.md)

  This explains the basic ideas around MemSave without diving into too many details.

- [Our paper (WANT@ICML'24)](https://openreview.net/pdf?id=KsUUzxUK7N) and it's [Poster](https://github.com/plutonium-239/memsave_torch/blob/main/memsave_poster.pdf)
  
  It is also available on [arXiv](https://arxiv.org/abs/2404.12406)

- [Documentation](https://memsave-torch.readthedocs.io/)
<!-- - [Link to more examples]()
- [Link to paper/experiments folder]()-->

## How to cite

If this package has benefited you at some point, consider citing

```bibtex
@inproceedings{
  bhatia2024lowering,
  title={Lowering PyTorch's Memory Consumption for Selective Differentiation},
  author={Samarth Bhatia and Felix Dangel},
  booktitle={2nd Workshop on Advancing Neural Network Training: Computational Efficiency, Scalability, and Resource Optimization (WANT@ICML 2024)},
  year={2024},
  url={https://openreview.net/forum?id=KsUUzxUK7N}
}
```
