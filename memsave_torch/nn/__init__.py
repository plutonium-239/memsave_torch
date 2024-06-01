"""Module containing implementations of memory saving neural network layers.

Currently implemented:
- Linear
- Conv2d
- BatchNorm2d
"""

import sys

import torch.nn as nn

from memsave_torch.nn import functional  # noqa: F401
from memsave_torch.nn.BatchNorm import MemSaveBatchNorm2d
from memsave_torch.nn.Conv1d import MemSaveConv1d
from memsave_torch.nn.Conv2d import MemSaveConv2d
from memsave_torch.nn.Dropout import MemSaveDropout
from memsave_torch.nn.LayerNorm import (
    MemSaveLayerNorm,
    MemSaveRMSLayerNorm,
    RMSLayerNorm,
)
from memsave_torch.nn.Linear import MemSaveLinear
from memsave_torch.nn.MaxPool import MemSaveMaxPool2d
from memsave_torch.nn.ReLU import MemSaveReLU

transformers_imported = False
if "transformers" in sys.modules:
    import transformers

    transformers_imported = True


def convert_to_memory_saving(
    model: nn.Module,
    linear=True,
    conv2d=True,
    conv1d=False,
    batchnorm2d=True,
    relu=True,
    maxpool2d=True,
    layernorm=True,
    dropout=True,
    verbose=False,
    clone_params=False,
) -> nn.Module:
    """Converts the given `model` to it's MemSave version, with options to choose which layer types to replace.

    The `clone_params` option should be used when you plan on using both models simultaneously. Otherwise,
    the grad accumulation for one model wll affect the other (since their weights are the same Tensor object).
    For an example, see tests/test_layers.py.

    Args:
        model (nn.Module): The input model
        linear (bool, optional): Whether to replace `nn.Linear` layers
        conv2d (bool, optional): Whether to replace `nn.Conv2d` layers
        conv1d (bool, optional): Whether to replace `nn.Conv1d` layers
        batchnorm2d (bool, optional): Whether to replace `nn.BatchNorm2d` layers
        relu (bool, optional): Whether to replace `nn.ReLU` layers
        maxpool2d (bool, optional): Whether to replace `nn.MaxPool2d` layers
        layernorm (bool, optional): Whether to replace `nn.LayerNorm` layers
        dropout (bool, optional): Whether to replace `nn.Dropout` layers
        verbose (bool, optional): Whether to print which layers were replaced
        clone_params (bool, optional): Whether to clone the layer parameters or use directly

    Returns:
        memsavemodel (nn.Module): The converted memory saving model
    """
    linear_cls = nn.Linear
    layernorm_cls = RMSLayerNorm
    if transformers_imported:
        linear_cls = (nn.Linear, transformers.Conv1D)
        layernorm_cls = (
            RMSLayerNorm,
            transformers.models.t5.modeling_t5.T5LayerNorm,
            transformers.models.mistral.modeling_mistral.MistralRMSNorm,
            transformers.models.llama.modeling_llama.LlamaRMSNorm,
            transformers.models.phi3.modeling_phi3.Phi3RMSNorm,
        )
    layers = [
        {
            "allowed": linear,
            "cls": linear_cls,
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
            "allowed": conv1d,
            "cls": nn.Conv1d,
            "convert_fn": MemSaveConv1d.from_nn_Conv1d,
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
        {
            "allowed": layernorm,
            "cls": layernorm_cls,
            "convert_fn": MemSaveRMSLayerNorm.from_existing,
        },
        {
            "allowed": dropout,
            "cls": nn.Dropout,
            "convert_fn": MemSaveDropout.from_nn_dropout,
        },
    ]

    import copy

    memsavemodel = copy.deepcopy(model)
    # using named_modules because it automatically iterates on Sequential/BasicBlock(resnet) etc.
    for name, layer in model.named_modules():
        for replacement in layers:
            if not replacement["allowed"] or not isinstance(layer, replacement["cls"]):
                continue
            if verbose:
                print(f"replaced {name}")
            if name == "":
                # In case a module is directly passed without wrapping sequential/moduledict
                return replacement["convert_fn"](layer)
            recursive_setattr(
                memsavemodel, name, replacement["convert_fn"](layer), clone_params
            )
    return memsavemodel


def recursive_setattr(obj: nn.Module, attr: str, value: nn.Module, clone_params: bool):
    """Taken from https://discuss.huggingface.co/t/how-can-i-replace-modules-in-a-pretrained-model/16338.

    Basically splits the full feature name (layer4.1.bn2) and recurses until non-iterable layer (conv2d/linear etc)

    Args:
        obj (nn.Module): Any module (the root of attr)
        attr (str): The dot-indexed name of the leaf layer to replace (i.e. layer.0.conv2)
        value (nn.Module): The module to replace the leaf with
        clone_params (bool): Whether to make a copy of the parameters or reuse them
    """
    attr_split = attr.split(".", 1)
    if len(attr_split) == 1:
        setattr(obj, attr_split[0], value)
        if clone_params:
            value.load_state_dict(value.state_dict())  # makes a copy
    else:
        recursive_setattr(
            getattr(obj, attr_split[0]), attr_split[1], value, clone_params
        )
