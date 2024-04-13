# `memsave_torch`: Lowering PyTorch's Memory Consumption for Selective Differentiation

This package offers drop-in implementations of PyTorch `nn.Module`s.
They are as fast as their built-in equivalents, but more memory-efficient whenever you want to compute gradients for a sub-set of parameters (i.e. some have `requires_grad=False`).
You can convert your neural network by calling the `memsave_torch.nn.convert_to_memory_saving` function.

Take a look at the [Basic Example](#basic-example) to see how it works.

Currently it supports the following layers:
- `memsave_torch.nn.MemSaveLinear`
- `memsave_torch.nn.MemSaveConv2d`
- `memsave_torch.nn.MemSaveConv1d`
- `memsave_torch.nn.MemSaveReLU`
- `memsave_torch.nn.MemSaveBatchNorm2d`
- `memsave_torch.nn.MemSaveLayerNorm`
- `memsave_torch.nn.MemSaveMaxPool2d`

Also, each layer has a `.from_nn_<layername>(layer)` function which allows to convert a single `torch.nn` layer into its memory-saving equivalent. (e.g. `MemSaveConv2d.from_nn_Conv2d`)

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
- [Link to documentation]()
- [Link to more examples]()
- [Link to paper/experiments folder]()
- [Writeup](memsave_torch/writeup.md)

## How to cite

If this package has benefited you at some point, consider citing

```bibtex

@article{
  TODO
}

```
