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


def convert_to_memory_saving(
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
    layers = [
        {
            "allowed": linear,
            "cls": nn.Linear,
            "convert_fn": MemSaveLinear.from_nn_Linear,
        },
        {"allowed": relu, "cls": nn.ReLU, "convert_fn": MemSaveReLU.from_nn_ReLU},
        {
            "allowed": maxpool2d,
            "cls": nn.MaxPool2d,
            "convert_fn": MemSaveMaxPool2d.from_nn_MaxPool2d,
        },
        {
            "allowed": conv2d,
            "cls": nn.Conv2d,
            "convert_fn": MemSaveConv2d.from_nn_Conv2d,
        },
        {
            "allowed": batchnorm2d,
            "cls": nn.BatchNorm2d,
            "convert_fn": MemSaveBatchNorm2d.from_nn_BatchNorm2d,
        },
        {
            "allowed": layernorm,
            "cls": nn.LayerNorm,
            "convert_fn": MemSaveLayerNorm.from_nn_LayerNorm,
        },
    ]

    import copy

    memsavemodel = copy.deepcopy(model)
    # using named_modules because it automatically iterates on Sequential/BasicBlock(resnet) etc.
    for name, layer in model.named_modules():
        for replacement in layers:
            if not replacement["allowed"] and isinstance(layer, replacement["cls"]):
                continue
            if verbose:
                print(f"replaced {name}")
            if name == "":
                # In case a module is directly passed without wrapping sequential/moduledict
                return replacement["convert_fn"](layer)
            recursive_setattr(memsavemodel, name, replacement["convert_fn"](layer))
    return memsavemodel


def recursive_setattr(obj: nn.Module, attr: str, value: nn.Module):
    """Taken from https://discuss.huggingface.co/t/how-can-i-replace-modules-in-a-pretrained-model/16338.

    Basically splits the full feature name (layer4.1.bn2) and recurses until non-iterable layer (conv2d/linear etc)

    Args:
        obj (nn.Module): Any module (the root of attr)
        attr (str): The dot-indexed name of the leaf layer to replace (i.e. layer.0.conv2)
        value (nn.Module): The module to replace the leaf with
    """
    attr_split = attr.split(".", 1)
    if len(attr) == 1:
        setattr(obj, attr_split[0], value)
    else:
        recursive_setattr(getattr(obj, attr_split[0]), attr_split[1], value)
