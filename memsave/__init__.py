"""Module containing implementations of memory saving neural network layers.

Currently implemented:
- Linear
- Conv2d
- BatchNorm2d
"""

from memsave.Linear import MemSaveLinear
from memsave.Conv2d import MemSaveConv2d
from memsave.BatchNorm import MemSaveBatchNorm2d
from memsave.MaxPool import MemSaveMaxPool2d
from memsave.ReLU import MemSaveReLU

import torch.nn as nn

class Identity(nn.Module):
	def forward(self, x):
		return x

def convert_to_memory_saving(
    model: nn.Module, linear=True, conv2d=True, batchnorm2d=True, relu=True, maxpool2d=True, verbose=False
) -> nn.Module:
    """Converts the given `model` to it's MemSave version, with options to choose which layer types to replace.
    
    Args:
        model (nn.Module): The input model
        linear (bool, optional): Whether to replace `nn.Linear` layers
        conv2d (bool, optional): Whether to replace `nn.Conv2d` layers
        batchnorm2d (bool, optional): Whether to replace `nn.BatchNorm2d` layers
        relu (bool, optional): Whether to replace `nn.ReLU` layers
        maxpool2d (bool, optional): Whether to replace `nn.MaxPool2d` layers
        verbose (bool, optional): Whether to print which layers were replaced
    
    Returns:
        memsavemodel (nn.Module): The converted memory saving model
    """
    import copy

    memsavemodel = copy.deepcopy(model)
    # using named_modules because it automatically iterates on Sequential/BasicBlock(resnet) etc.
    for name, layer in model.named_modules():
        if relu and isinstance(layer, nn.ReLU):
            if verbose: print(f"replaced {name}")
            recursive_setattr(memsavemodel, name, MemSaveReLU.from_nn_ReLU(layer))
        if maxpool2d and isinstance(layer, nn.MaxPool2d):
            if verbose: print(f"replaced {name}")
            recursive_setattr(memsavemodel, name, MemSaveMaxPool2d.from_nn_MaxPool2d(layer))
        if linear and isinstance(layer, nn.Linear):
            if verbose: print(f"replaced {name}")
            recursive_setattr(memsavemodel, name, MemSaveLinear.from_nn_Linear(layer))
        if conv2d and isinstance(layer, nn.Conv2d):
            if verbose: print(f"replaced {name}")
            recursive_setattr(memsavemodel, name, MemSaveConv2d.from_nn_Conv2d(layer))
        if batchnorm2d and isinstance(layer, nn.BatchNorm2d):
            if verbose: print(f"replaced {name}")
            recursive_setattr(
                memsavemodel, name, MemSaveBatchNorm2d.from_nn_BatchNorm2d(layer)
            )

    return memsavemodel


def recursive_setattr(obj: nn.Module, attr: str, value: nn.Module):
    """Taken from https://discuss.huggingface.co/t/how-can-i-replace-modules-in-a-pretrained-model/16338.

    Basically splits the full feature name (layer4.1.bn2) and recurses until non-iterable layer (conv2d/linear etc)

    Args:
        obj (nn.Module):
        attr (str): Description
        value (nn.Module): Description
    """
    attr = attr.split(".", 1)
    if len(attr) == 1:
        setattr(obj, attr[0], value)
    else:
        recursive_setattr(getattr(obj, attr[0]), attr[1], value)
