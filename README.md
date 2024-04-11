# `memsave_torch`: Lowering PyTorch's Memory Consumption for Selective Differentiation

This package offers drop-in implementations of PyTorch `nn.Module`s.
They are as fast as their built-in equivalents, but more memory-efficient whenever you want to compute gradients for a sub-set of parameters (i.e. some have `requires_grad=False`).
You can convert your neural network by calling our converter.

Take a look at the 'Basic example' how it works.

Currently it supports the following layers:

- `memsave_torch.nn.MemSaveConv2d`
- ...

Also, each layer has a `.from_nn(layer)` function which allows to convert a `torch.nn.Layer` into its memory-saving equivalent.

## Installation

Install (editable):
```bash
pip install -e .
```

## Basic example

```python
X, y = ...

model = Sequential(...)
loss_func = MSELoss()

# replace all available layers with mem-saving layers
model = convert(...)

# same as if you were using the original model
loss = loss_func(model(X), y)
```

## Further reading

- Link to documentation

- Link to more examples

- Link to paper/experiments folder

[Writeup](memsave_torch/writeup.md)

## How to cite

If this package has benefited you at some point, consider citing

```bibtex

@article{
  TODO
}

```
