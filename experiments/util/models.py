"""Utility file defining the various models, add more in the conv_model_fns dict."""

import itertools
import math
import warnings
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torchvision.models as tvm
from torch.nn import (
    BatchNorm2d,
    Conv2d,
    Flatten,
    Linear,
    MaxPool2d,
    Module,
    ReLU,
    Sequential,
    Transformer,
    Unfold,
    functional,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForPreTraining,
    AutoModelForSeq2SeqLM,
    BartForConditionalGeneration,
)
from transformers import logging as tf_logging
from transformers import utils as tf_utils

from memsave_torch.nn import (
    MemSaveBatchNorm2d,
    convert_to_memory_saving,
)

num_classes: int = 1


def prefix_in_pairs(prefix: str, it: List[str]) -> List[str]:
    """Prefixes the given `prefix` after each entry of the list `it`.

    Example:
        >>> models = ['resnet101', 'convnext']
        >>> prefix_in_pairs('memsave_', models)
        ['resnet101', 'memsave_resnet101', 'convnext', 'memsave_convnext']

    Args:
        prefix (str): Prefix to be added
        it (List[str]): The list to be prefixed

    Returns:
        List[str]: The output iterable with items prefixed in pairs
    """
    new_it = [[m, f"{prefix}{m}"] for m in it]
    new_it = list(itertools.chain.from_iterable(new_it))  # flatten list of lists
    return new_it


def convert_to_memory_saving_defaultsoff(
    model: Module,
    conv2d=False,
    conv1d=False,
    conv3d=False,
    batchnorm2d=False,
    relu=False,
    maxpool2d=False,
    dropout=False,
) -> Module:
    """Extension of the `convert_to_memory_saving` function with all defaults as off

    Args:
        model (Module): Input model
        conv2d (bool, optional): Whether to replace conv2d layers
        conv1d (bool, optional): Whether to replace conv1d layers
        conv3d (bool, optional): Whether to replace conv3d layers
        batchnorm2d (bool, optional): Whether to replace batchnorm2d layers
        relu (bool, optional): Whether to replace relu layers
        maxpool2d (bool, optional): Whether to replace maxpool2d layers
        dropout (bool, optional): Whether to replace dropout layers

    Returns:
        Module: The converted memory saving model
    """
    return convert_to_memory_saving(
        model,
        conv2d=conv2d,
        conv1d=conv1d,
        conv3d=conv3d,
        batchnorm2d=batchnorm2d,
        relu=relu,
        maxpool2d=maxpool2d,
        dropout=dropout,
    )


def get_transformers_config(model_name: str) -> AutoConfig:
    """Get the config for the given `model_name` from huggingface transformers. Handles memsave_ as well.

    Args:
        model_name (str): Model name

    Returns:
        AutoConfig: Config for given model
    """
    if model_name.startswith("memsave_"):
        model_name = model_name.split("memsave_")[1]
    props = hf_transformers_models_map[model_name]
    return AutoConfig.from_pretrained(props.hf_name, **props.extra_kwargs)


def get_arch_models(arch: str) -> Tuple[Dict[str, Callable], Any]:
    """Get the dict of all defined functions for an architecture

    Args:
        arch (str): The architecture

    Returns:
        Tuple[Dict[str, Callable], Any]: Dict of all defined functions

    Raises:
        ValueError: Invalid architecture
    """
    if arch == "conv":
        return conv_model_fns, conv_input_shape
    if arch == "transformer":
        return transformer_model_fns, transformer_input_shape
    if arch == "linear":
        return linear_model_fns, linear_input_shape
    raise ValueError(f"arch={arch} not in allowed architectures")


def set_BN_to_eval(model: Module) -> Module:
    """Sets all BatchNorm layers in the input `model` to eval mode (i.e. bn.eval()).

    Args:
        model (Module): Input model

    Returns:
        Module: Model with BN layers in eval mode
    """
    known_bn_layers = (BatchNorm2d, MemSaveBatchNorm2d)
    for layer in model.modules():
        if isinstance(layer, known_bn_layers):
            layer.eval()
    return model


# CONV
conv_input_shape: Tuple[int, int, int] = (1, 1, 1)


def _conv_model1() -> Module:
    return Sequential(
        Conv2d(conv_input_shape[0], 64, kernel_size=3, padding=1, bias=False),
        MaxPool2d(kernel_size=3, stride=2, padding=1),
        ReLU(),
        *[Conv2d(64, 64, kernel_size=3, padding=1, bias=False) for _ in range(10)],
        MaxPool2d(kernel_size=4, stride=4, padding=1),
        Flatten(start_dim=1, end_dim=-1),
        Linear(conv_input_shape[1] * conv_input_shape[2], num_classes),
    )  # (H/8)*(W/8)*64 (filters) -> / 8 because maxpool


def _convrelu_model1() -> Module:
    return Sequential(
        Conv2d(conv_input_shape[0], 64, kernel_size=3, padding=1, bias=False),
        MaxPool2d(kernel_size=3, stride=2, padding=1),
        ReLU(),
        *list(
            itertools.chain.from_iterable(
                [
                    (Conv2d(64, 64, kernel_size=3, padding=1, bias=False), ReLU())
                    for _ in range(10)
                ]
            )
        ),
        MaxPool2d(kernel_size=4, stride=4, padding=1),
        Flatten(start_dim=1, end_dim=-1),
        Linear(conv_input_shape[1] * conv_input_shape[2], num_classes),
    )  # (H/8)*(W/8)*64 (filters) -> / 8 because maxpool


def _convrelupool_model1(num_blocks=5) -> Module:
    assert min(conv_input_shape[1:]) > 2 ** (num_blocks + 1)
    return Sequential(
        Conv2d(conv_input_shape[0], 64, kernel_size=3, padding=1, bias=False),
        MaxPool2d(kernel_size=3, stride=2, padding=1),
        ReLU(),
        *list(
            itertools.chain.from_iterable(
                [
                    (
                        Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                        ReLU(),
                        MaxPool2d(kernel_size=3, stride=2, padding=1),
                    )
                    for _ in range(num_blocks)
                ]
            )
        ),
        Flatten(start_dim=1, end_dim=-1),
        Linear(
            64
            * math.ceil(conv_input_shape[1] / 2 ** (num_blocks + 1))
            * math.ceil(conv_input_shape[2] / 2 ** (num_blocks + 1)),
            num_classes,
        ),
    )


detection_models = [
    "fasterrcnn_resnet50_fpn_v2",
    "retinanet_resnet50_fpn_v2",
    "ssdlite320_mobilenet_v3_large",
]
detection_models = prefix_in_pairs("memsave_", detection_models)
segmentation_models = ["deeplabv3_resnet101", "fcn_resnet101"]
segmentation_models = prefix_in_pairs("memsave_", segmentation_models)
models_without_norm = ["deepmodel", "vgg16"]
models_without_norm = prefix_in_pairs("memsave_", models_without_norm)

conv_model_fns = {
    "deepmodel": _conv_model1,
    "memsave_deepmodel": lambda: convert_to_memory_saving(_conv_model1()),
    "deeprelumodel": _convrelu_model1,
    "memsave_deeprelumodel": lambda: convert_to_memory_saving(_convrelu_model1()),
    "deeprelupoolmodel": _convrelupool_model1,
    "memsave_deeprelupoolmodel": lambda: convert_to_memory_saving(
        _convrelupool_model1()
    ),
    "alexnet": tvm.alexnet,
    "memsave_alexnet": lambda: convert_to_memory_saving(tvm.alexnet()),
    "convnext_base": tvm.convnext_base,
    "memsave_convnext_base": lambda: convert_to_memory_saving(tvm.convnext_base()),
    "resnet101": tvm.resnet101,
    "memsave_resnet101": lambda: convert_to_memory_saving(tvm.resnet101()),
    "vgg16": tvm.vgg16,
    "memsave_vgg16": lambda: convert_to_memory_saving(tvm.vgg16()),
    "resnet18": tvm.resnet18,
    "memsave_resnet18": lambda: convert_to_memory_saving(tvm.resnet18()),
    "fasterrcnn_resnet50_fpn_v2": tvm.detection.fasterrcnn_resnet50_fpn_v2,
    "memsave_fasterrcnn_resnet50_fpn_v2": lambda: convert_to_memory_saving(
        tvm.detection.fasterrcnn_resnet50_fpn_v2()
    ),
    "retinanet_resnet50_fpn_v2": tvm.detection.retinanet_resnet50_fpn_v2,
    "memsave_retinanet_resnet50_fpn_v2": lambda: convert_to_memory_saving(
        tvm.detection.retinanet_resnet50_fpn_v2()
    ),
    "ssdlite320_mobilenet_v3_large": tvm.detection.ssdlite320_mobilenet_v3_large,
    "memsave_ssdlite320_mobilenet_v3_large": lambda: convert_to_memory_saving(
        tvm.detection.ssdlite320_mobilenet_v3_large()
    ),
    "deeplabv3_resnet101": tvm.segmentation.deeplabv3_resnet101,
    "memsave_deeplabv3_resnet101": lambda num_classes: convert_to_memory_saving(
        tvm.segmentation.deeplabv3_resnet101(num_classes=num_classes)
    ),
    "fcn_resnet101": tvm.segmentation.fcn_resnet101,
    "memsave_fcn_resnet101": lambda num_classes: convert_to_memory_saving(
        tvm.segmentation.fcn_resnet101(num_classes=num_classes)
    ),
    "efficientnet_v2_l": tvm.efficientnet_v2_l,
    "memsave_efficientnet_v2_l": lambda: convert_to_memory_saving(
        tvm.efficientnet_v2_l()
    ),
    "mobilenet_v3_large": tvm.mobilenet_v3_large,
    "memsave_mobilenet_v3_large": lambda: convert_to_memory_saving(
        tvm.mobilenet_v3_large()
    ),
    "resnext101_64x4d": tvm.resnext101_64x4d,
    "memsave_resnext101_64x4d": lambda: convert_to_memory_saving(
        tvm.resnext101_64x4d()
    ),
    # For paper
    "memsave_resnet101_conv": lambda: convert_to_memory_saving_defaultsoff(
        tvm.resnet101(), conv2d=True
    ),
    "memsave_resnet101_conv+relu+bn": lambda: convert_to_memory_saving_defaultsoff(
        tvm.resnet101(), conv2d=True, relu=True, batchnorm2d=True
    ),
    "memsave_resnet101_conv_full": lambda: convert_to_memory_saving(tvm.resnet101()),
}


class SegmentationLossWrapper(Module):
    """Small wrapper around a loss to support interop with existing measurement code"""

    def __init__(self, loss_fn_orig) -> None:
        """Init

        Attributes:
            loss_fn: A function which returns a loss nn.Module
        """
        super().__init__()
        self.loss_fn = loss_fn_orig()

    def forward(self, x, y):
        """Forward

        Args:
            x: x
            y: y

        Returns:
           output: loss
        """
        return self.loss_fn(x["out"], y)


class DetectionLossWrapper(Module):
    """Small wrapper around a loss to support interop with existing measurement code"""

    def __init__(self) -> None:
        """Init"""
        super().__init__()

    def forward(self, loss_dict):
        """Forward

        Args:
            loss_dict: loss_dict

        Returns:
            output: loss
        """
        return sum(loss_dict.values())


# TRANSFORMER
transformer_input_shape: Tuple[int, int] = (1, 1)  # (vocab_dim, embed_dim)


class _HF_model:
    def __init__(
        self,
        hf_name: str,
        extra_kwargs: Dict[str, Any],
        model_cls: Any = AutoModelForCausalLM,
        lm_head_name: Optional[str] = None,
    ) -> None:
        self.hf_name = hf_name
        self.extra_kwargs = extra_kwargs
        if self.extra_kwargs is None:
            self.extra_kwargs = {}
        self.model_cls = model_cls
        self.lm_head_name = lm_head_name


tf_logging.disable_progress_bar()
tf_logging.set_verbosity_error()
tf_utils.logging.captureWarnings(True)


hf_transformers_models_map = {
    "gpt2": _HF_model("gpt2", {}, lm_head_name="lm_head"),
    "vit": _HF_model("facebook/vit-mae-base", {}, AutoModelForPreTraining),
    "bert": _HF_model(
        "google-bert/bert-base-uncased",
        {"is_decoder": True},
        lm_head_name="cls.predictions.decoder",
    ),
    "bart": _HF_model(
        "facebook/bart-base", {}, BartForConditionalGeneration, "lm_head"
    ),
    "roberta": _HF_model(
        "FacebookAI/roberta-base", {"is_decoder": True}, lm_head_name="lm_head.decoder"
    ),
    "t5": _HF_model("google-t5/t5-base", {}, AutoModelForSeq2SeqLM, "lm_head"),
    "flan-t5": _HF_model("google/flan-t5-base", {}, AutoModelForSeq2SeqLM, "lm_head"),
    "xlm-roberta": _HF_model(
        "FacebookAI/xlm-roberta-base", {}, AutoModelForMaskedLM, "lm_head.decoder"
    ),
    "mistral-7b": _HF_model(
        "mistralai/Mistral-7B-v0.1",
        {"torch_dtype": torch.bfloat16},
        lm_head_name="lm_head",
    ),
    "llama3-8b": _HF_model(
        "meta-llama/Meta-Llama-3-8B",
        {"torch_dtype": torch.bfloat16},
        lm_head_name="lm_head",
    ),
    "phi3-4b": _HF_model(
        "microsoft/Phi-3-mini-4k-instruct",
        {"torch_dtype": torch.bfloat16},
        lm_head_name="lm_head",
    ),
}
hf_transformers_models = list(hf_transformers_models_map.keys())
hf_transformers_models = prefix_in_pairs("memsave_", hf_transformers_models)

transformer_model_fns = {
    "transformer": lambda: TorchTransformer(),
    "memsave_transformer": lambda: convert_to_memory_saving(TorchTransformer()),
}

fused = lambda fn, name, kwargs: convert_to_memory_saving(  # noqa: E731
    fn(name, **kwargs)
)

for m in hf_transformers_models:
    if m in transformer_model_fns:
        continue
    # Can't use lambdas in loops :')
    if not m.startswith("memsave_"):
        props = hf_transformers_models_map[m]
        transformer_model_fns[m] = partial(
            props.model_cls.from_pretrained, props.hf_name, **props.extra_kwargs
        )
    else:
        props = hf_transformers_models_map[m.split("memsave_", 1)[1]]
        transformer_model_fns[m] = partial(
            fused, props.model_cls.from_pretrained, props.hf_name, props.extra_kwargs
        )


class TorchTransformer(Module):
    """Small model to wrap `torch.nn.Transformer`"""

    def __init__(self) -> None:
        """Init"""
        super().__init__()
        self.transformer = Transformer(
            d_model=transformer_input_shape[1], batch_first=True
        )
        self.pred = Linear(transformer_input_shape[1], transformer_input_shape[0])

    def forward(self, x):
        """Forward

        Args:
            x: x

        Returns:
            output: model output
        """
        out = self.transformer.decoder(x, self.transformer.encoder(x))
        return self.pred(out).permute(0, 2, 1)


class TransformersModelWrapper(Module):
    """Small wrapper around `transformers` models to support interop with existing measurement code"""

    def __init__(self, model_fn, model_name) -> None:
        """Init"""
        super().__init__()
        with warnings.catch_warnings():
            # hf does not keep quiet sometimes even when transformers.logging is set to errors only
            # https://github.com/huggingface/transformers/issues/30618
            warnings.simplefilter("ignore")
            self.model = model_fn()
        self.dec = self.model.config.is_encoder_decoder
        self.model_name = model_name
        model_name_pure = model_name
        if model_name.startswith("memsave_"):
            model_name_pure = model_name.split("memsave_")[1]
        self.lm_head_name = hf_transformers_models_map[model_name_pure].lm_head_name

        self.cache_kw = {"use_cache": False}
        if any("ForMaskedLM" in a for a in self.model.config.architectures):
            self.cache_kw = {}

    def forward(self, x):
        """Forward

        Args:
            x: x

        Returns:
            output: model output
        """
        if self.model.dtype != torch.float32:
            x = x.to(self.model.dtype)
            # HF takes care of converting logits to float32
        if self.dec:
            out = self.model(inputs_embeds=x, decoder_inputs_embeds=x, **self.cache_kw)
        else:
            out = self.model(inputs_embeds=x, **self.cache_kw)
        return out.logits.permute(0, 2, 1)


# VLM
class VLM(Module):
    """Small wrapper for making a VLM model with transformer llm and conv/transformer vision model"""

    def __init__(
        self,
        vision_model_name: str,
        vision_model_arch: str,
        llm_name: str,
        nc: int = 1000,
    ) -> None:
        """Init"""
        super().__init__()
        self.vision_model_name = vision_model_name
        self.vm_arch = vision_model_arch
        self.llm_name = llm_name
        model_fns, input_shape = get_arch_models(vision_model_arch)
        if vision_model_arch == "conv":
            assert vision_model_name in segmentation_models
        self.vm = model_fns[vision_model_name]()
        self.llm = TransformersModelWrapper(transformer_model_fns[llm_name], llm_name)
        vision_final_dim = 3 * 16 * 16 if vision_model_arch == "transformer" else nc
        self.proj = Linear(vision_final_dim, self.llm.model.config.hidden_size)
        self.patchify = Unfold(kernel_size=16, stride=16)

    def forward(self, x):
        """Forward through vlm

        Args:
            x: x

        Returns:
            output: model output
        """
        if self.vm_arch == "transformer" and self.vm.config.image_size != x.shape[-1]:
            x = functional.interpolate(
                x, size=self.vm.config.image_size, mode="bicubic"
            )
        x = self.vm(x)
        if self.vm_arch == "conv":
            import ipdb

            ipdb.set_trace()
            x = self.patchify(x["out"]).permute(0, 2, 1)
            # [B, nc*n_patches, patch_size**2]
        else:
            x = x.logits
        x = self.proj(x)
        # [B, patch_size**2, llm_hidden]
        return self.llm(x)


# LINEAR
linear_input_shape: int = 1


def _linear_model1() -> Module:
    return Sequential(
        Linear(linear_input_shape, 1024),
        *[Linear(1024, 1024) for _ in range(12)],
        Linear(1024, num_classes),
    )


linear_model_fns = {
    "deeplinearmodel": _linear_model1,
    # Doesn't do anything, just kept for consistency:
    "memsave_deeplinearmodel": lambda: convert_to_memory_saving(_linear_model1()),
}
