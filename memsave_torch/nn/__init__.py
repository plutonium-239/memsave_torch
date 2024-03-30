"""Module containing implementations of memory saving neural network layers.

Currently implemented:
- Linear
- Conv2d
- BatchNorm2d
"""

import torch.nn as nn

from memsave_torch.nn.BatchNorm import MemSaveBatchNorm2d
from memsave_torch.nn.Conv2d import MemSaveConv2d
from memsave_torch.nn.LayerNorm import MemSaveLayerNorm
from memsave_torch.nn.Linear import MemSaveLinear
from memsave_torch.nn.MaxPool import MemSaveMaxPool2d
from memsave_torch.nn.ReLU import MemSaveReLU


def convert_to_memory_saving(  # noqa: C901
    model: nn.Module,
    linear=True,
    conv2d=True,
    batchnorm2d=True,
    relu=True,
    maxpool2d=True,
    layernorm=True,
    verbose=False,
) -> nn.Module:
    """Converts the given `model` to it's MemSave version, with options to choose which layer types to replace.

    Args:
        model (nn.Module): The input model
        linear (bool, optional): Whether to replace `nn.Linear` layers
        conv2d (bool, optional): Whether to replace `nn.Conv2d` layers
        batchnorm2d (bool, optional): Whether to replace `nn.BatchNorm2d` layers
        relu (bool, optional): Whether to replace `nn.ReLU` layers
        maxpool2d (bool, optional): Whether to replace `nn.MaxPool2d` layers
        layernorm (bool, optional): Whether to replace `nn.LayerNorm` layers
        verbose (bool, optional): Whether to print which layers were replaced

    Returns:
        memsavemodel (nn.Module): The converted memory saving model
    """
    import copy

    memsavemodel = copy.deepcopy(model)
    # using named_modules because it automatically iterates on Sequential/BasicBlock(resnet) etc.
    for name, layer in model.named_modules():
        if relu and isinstance(layer, nn.ReLU):
            if verbose:
                print(f"replaced {name}")
            recursive_setattr(memsavemodel, name, MemSaveReLU.from_nn_ReLU(layer))
        if maxpool2d and isinstance(layer, nn.MaxPool2d):
            if verbose:
                print(f"replaced {name}")
            recursive_setattr(
                memsavemodel, name, MemSaveMaxPool2d.from_nn_MaxPool2d(layer)
            )
        if linear and isinstance(layer, nn.Linear):
            if verbose:
                print(f"replaced {name}")
            recursive_setattr(memsavemodel, name, MemSaveLinear.from_nn_Linear(layer))
        if conv2d and isinstance(layer, nn.Conv2d):
            if verbose:
                print(f"replaced {name}")
            recursive_setattr(memsavemodel, name, MemSaveConv2d.from_nn_Conv2d(layer))
        if batchnorm2d and isinstance(layer, nn.BatchNorm2d):
            if verbose:
                print(f"replaced {name}")
            recursive_setattr(
                memsavemodel, name, MemSaveBatchNorm2d.from_nn_BatchNorm2d(layer)
            )
        if layernorm and isinstance(layer, nn.LayerNorm):
            if verbose:
                print(f"replaced {name}")
            recursive_setattr(
                memsavemodel, name, MemSaveLayerNorm.from_nn_LayerNorm(layer)
            )

    return memsavemodel


def recursive_setattr(obj: nn.Module, attr: str, value: nn.Module):
    """Taken from https://discuss.huggingface.co/t/how-can-i-replace-modules-in-a-pretrained-model/16338.

    Basically splits the full feature name (layer4.1.bn2) and recurses until non-iterable layer (conv2d/linear etc)

    Args:
        obj (nn.Module): Any module (the root of attr)
        attr (str): The dot-indexed name of the leaf layer to replace (i.e. layer.0.conv2)
        value (nn.Module): The module to replace the leaf with
    """
    attr = attr.split(".", 1)
    if len(attr) == 1:
        setattr(obj, attr[0], value)
    else:
        recursive_setattr(getattr(obj, attr[0]), attr[1], value)
