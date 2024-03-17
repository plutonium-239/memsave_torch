from typing import Tuple

from torch.nn import Conv2d, Flatten, Linear, MaxPool2d, Module, ReLU, Sequential
import torchvision.models as tvm
from memsave import (
    MemSaveLinear,
    MemSaveConv2d,
    MemSaveBatchNorm2d,
    convert_to_memory_saving,
)
import itertools
import math

num_classes: int = 1

def prefix_memsave_in(models):
    models = [[m, f"memsave_{m}"] for m in models]  # add memsave versions for each model
    models = list(itertools.chain.from_iterable(models))  # flatten list of lists
    return models

def convert_to_memory_saving_defaultsoff(
    model: Module, linear=False, conv2d=False, batchnorm2d=False, relu=False, maxpool2d=False
) -> Module:
    return convert_to_memory_saving(model, linear, conv2d, batchnorm2d, relu, maxpool2d)

# CONV
conv_input_shape: Tuple[int, int, int] = (1, 1, 1)


def conv_model1() -> Module:
    return Sequential(
        Conv2d(conv_input_shape[0], 64, kernel_size=3, padding=1, bias=False),
        MaxPool2d(kernel_size=3, stride=2, padding=1),
        ReLU(),
        *[Conv2d(64, 64, kernel_size=3, padding=1, bias=False) for _ in range(10)],
        MaxPool2d(kernel_size=4, stride=4, padding=1),
        Flatten(start_dim=1, end_dim=-1),
        Linear(conv_input_shape[1] * conv_input_shape[2], num_classes),
    )  # (H/8)*(W/8)*64 (filters) -> / 8 because maxpool


def conv_model2() -> Module:
    return Sequential(
        MemSaveConv2d(conv_input_shape[0], 64, kernel_size=3, padding=1, bias=False),
        MaxPool2d(kernel_size=3, stride=2, padding=1),
        ReLU(),
        *[
            MemSaveConv2d(64, 64, kernel_size=3, padding=1, bias=False)
            for _ in range(10)
        ],
        MaxPool2d(kernel_size=4, stride=4, padding=1),
        Flatten(start_dim=1, end_dim=-1),
        MemSaveLinear(conv_input_shape[1] * conv_input_shape[2], num_classes),
    )

def convrelu_model1() -> Module:
    return Sequential(
        Conv2d(conv_input_shape[0], 64, kernel_size=3, padding=1, bias=False),
        MaxPool2d(kernel_size=3, stride=2, padding=1),
        ReLU(),
        *list(itertools.chain.from_iterable([(
            Conv2d(64, 64, kernel_size=3, padding=1, bias=False), 
            ReLU()
            ) for _ in range(10)])),
        MaxPool2d(kernel_size=4, stride=4, padding=1),
        Flatten(start_dim=1, end_dim=-1),
        Linear(conv_input_shape[1] * conv_input_shape[2], num_classes),
    )  # (H/8)*(W/8)*64 (filters) -> / 8 because maxpool

def convrelupool_model1(num_blocks=5) -> Module:
    assert min(conv_input_shape[1:]) > 2**(num_blocks+1)
    return Sequential(
        Conv2d(conv_input_shape[0], 64, kernel_size=3, padding=1, bias=False),
        MaxPool2d(kernel_size=3, stride=2, padding=1),
        ReLU(),
        *list(itertools.chain.from_iterable([(
            Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            ReLU(),
            MaxPool2d(kernel_size=3, stride=2, padding=1),
            ) for _ in range(num_blocks)])),
        Flatten(start_dim=1, end_dim=-1),
        Linear(64 * math.ceil(conv_input_shape[1]/2**(num_blocks+1)) 
            * math.ceil(conv_input_shape[2]/2**(num_blocks+1)), num_classes),
    )

detection_models = ["fasterrcnn_resnet50_fpn_v2", "retinanet_resnet50_fpn_v2"]
detection_models = prefix_memsave_in(detection_models)
segmentation_models = ["deeplabv3_resnet101", "fcn_resnet101"]
segmentation_models = prefix_memsave_in(segmentation_models)

conv_model_fns = {
    "deepmodel": conv_model1,
    "memsave_deepmodel": conv_model2,
    "deeprelumodel": convrelu_model1,
    "memsave_deeprelumodel": lambda: convert_to_memory_saving(convrelu_model1()),
    "deeprelupoolmodel": convrelupool_model1,
    "memsave_deeprelupoolmodel": lambda: convert_to_memory_saving(convrelupool_model1()),
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
    "memsave_fasterrcnn_resnet50_fpn_v2": lambda: convert_to_memory_saving(tvm.detection.fasterrcnn_resnet50_fpn_v2()),
    "retinanet_resnet50_fpn_v2": tvm.detection.retinanet_resnet50_fpn_v2,
    "memsave_retinanet_resnet50_fpn_v2": lambda: convert_to_memory_saving(tvm.detection.retinanet_resnet50_fpn_v2()),
    "ssdlite320_mobilenet_v3_large": tvm.detection.ssdlite320_mobilenet_v3_large,
    "memsave_ssdlite320_mobilenet_v3_large": lambda: convert_to_memory_saving(tvm.detection.ssdlite320_mobilenet_v3_large()),
    "deeplabv3_resnet101": tvm.segmentation.deeplabv3_resnet101,
    "memsave_deeplabv3_resnet101": lambda num_classes: convert_to_memory_saving(tvm.segmentation.deeplabv3_resnet101(num_classes=num_classes)),
    "fcn_resnet101": tvm.segmentation.fcn_resnet101,
    "memsave_fcn_resnet101": lambda num_classes: convert_to_memory_saving(tvm.segmentation.fcn_resnet101(num_classes=num_classes)),
    "efficientnet_v2_l": tvm.efficientnet_v2_l,
    "memsave_efficientnet_v2_l": lambda: convert_to_memory_saving(tvm.efficientnet_v2_l()),
    "mobilenet_v3_large": tvm.mobilenet_v3_large,
    "memsave_mobilenet_v3_large": lambda: convert_to_memory_saving(tvm.mobilenet_v3_large()),
    "resnext101_64x4d": tvm.resnext101_64x4d,
    "memsave_resnext101_64x4d": lambda: convert_to_memory_saving(tvm.resnext101_64x4d()),

    "memsave_resnet101_conv": lambda: convert_to_memory_saving_defaultsoff(tvm.resnet101(), conv2d=True),
    "memsave_resnet101_conv+relu+bn": lambda: convert_to_memory_saving_defaultsoff(tvm.resnet101(), conv2d=True, relu=True, batchnorm2d=True),
    "memsave_resnet101_conv_full": lambda: convert_to_memory_saving(tvm.resnet101()),
}

class SegmentationLossWrapper(Module):
    """Small wrapper around a loss to support interop with existing measurement code
    
    Attributes:
        loss_fn: A function which returns a loss nn.Module
    """
    
    def __init__(self, loss_fn_orig) -> None:
        super().__init__()
        self.loss_fn = loss_fn_orig()

    def forward(self, x, y):
        return self.loss_fn(x['out'], y)

class DetectionLossWrapper(Module):
    """Small wrapper around a loss to support interop with existing measurement code
    
    Attributes:
        loss_fn: A function which returns a loss nn.Module
    """
    
    def __init__(self) -> None:
        super().__init__()

    def forward(self, loss_dict):
        return sum(loss_dict.values())

# LINEAR
linear_input_shape: int = 1


def linear_model1() -> Module:
    return Sequential(
        Linear(linear_input_shape, 1024),
        *[Linear(1024, 1024) for _ in range(12)],
        Linear(1024, num_classes),
    )


def linear_model2() -> Module:
    return Sequential(
        MemSaveLinear(linear_input_shape, 1024),
        *[MemSaveLinear(1024, 1024) for _ in range(12)],
        MemSaveLinear(1024, num_classes),
    )


linear_model_fns = {
    "deeplinearmodel": linear_model1,
    "memsave_deeplinearmodel": linear_model2,
}
