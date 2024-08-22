"""Contains a class to perform run time and memory measurements of forward/backward."""

import gc
from time import sleep
from typing import Callable, Dict, List, Optional, Tuple

import torch
from codetiming import Timer
from memory_profiler import memory_usage
from torch import Tensor, cuda, device

# from torch.profiler import profile, record_function, ProfilerActivity
from torch.nn import (
    BatchNorm1d,
    BatchNorm2d,
    BatchNorm3d,
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
    Embedding,
    LayerNorm,
    Linear,
    Module,
    Parameter,
)
from torchvision.models.convnext import LayerNorm2d
from transformers import Conv1D

from memsave_torch.nn.Conv2d import MemSaveConv2d
from memsave_torch.nn.Linear import MemSaveLinear


def maybe_synchronize(dev: device):
    """Synchronize CUDA kernels if device is GPU.

    Args:
        dev: PyTorch device.
    """
    if "cuda" in str(dev):
        cuda.synchronize()


class Measurement:
    """Base class for measurements. This is meant to be subclassed and extended."""

    def __init__(
        self,
        model_fn: Callable[[], Module],
        loss_fn: Callable[[], Module],
        x: Tensor,
        y: Tensor,
        dev: device,
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ):
        """Store the model, loss function, inputs, labels, and the device.

        Args:
            model_fn: A function that returns a model.
            loss_fn: A function that returns a loss function.
            x: The input tensor.
            y: The output tensor.
            dev: The device to measure run time on.
            targets: Targets in case of detection model.
        """
        self.model_fn = model_fn
        self.loss_fn = loss_fn
        self.x = x
        self.y = y
        self.dev = dev
        self.targets = targets

    def set_up(
        self,
        synchronize: bool = True,
        grad_linear_weights: bool = True,
        grad_linear_bias: bool = True,
        grad_conv_weights: bool = True,
        grad_conv_bias: bool = True,
        grad_norm_weights: bool = True,
        grad_norm_bias: bool = True,
        grad_input: bool = False,
        grad_embed_weights: bool = False,
        surgical_first: bool = False,
        surgical_last: bool = False,
    ) -> Tuple[
        Module,
        Module,
        Tensor,
        Tensor,
        Optional[List[Dict[str, Tensor]]],
        List[Tensor],
        List[Tensor],
    ]:
        """Initialize model and loss function, load to device (including data).

        Syncs CUDA threads if the device is a GPU to avoid leaking run time
        of this function into the measurement.

        Args:
            synchronize (bool, optional): Whether to synchronize CUDA threads after loading the
                model, loss function, and data to the device. Default: `True`.
            grad_linear_weights (bool, optional): Whether to compute the gradient of the linear
                layer weights. Default: `True`.
            grad_linear_bias (bool, optional): Whether to compute the gradient of the linear
                layer bias. Default: `True`.
            grad_conv_weights (bool, optional): Whether to compute the gradient of the convolution
                layer weights. Default: `True`.
            grad_conv_bias (bool, optional): Whether to compute the gradient of the convolution
                layer bias. Default: `True`.
            grad_norm_weights (bool, optional): Whether to compute the gradient of the normalization
                layer weights. Default: `True`.
            grad_norm_bias (bool, optional): Whether to compute the gradient of the normalization
                layer bias. Default: `True`.
            grad_input (bool, optional): Whether to compute the gradient of the input. Default: `False`.
            grad_embed_weights (bool, optional): Whether to compute the gradient of the embedding
                layer weights. Default: `True`.
            surgical_first (bool, optional): Corresponds to computing gradient only for the first quarter of layers
            surgical_last (bool, optional): Corresponds to computing gradient only for the last quarter of layers

        Returns:
            The model, loss function, input tensor, and output tensor. All are loaded
            to the specified device.
        """
        model = self.model_fn().to(self.dev)
        loss_fn = self.loss_fn().to(self.dev)
        x = self.x.clone().detach().to(self.dev)
        y = self.y.clone().detach().to(self.dev)
        if self.targets is not None:
            targets = [
                {k: v.clone().detach().to(self.dev) for k, v in di.items()}
                for di in self.targets
            ]
        else:
            targets = None

        if surgical_first or surgical_last:
            leafs, no_leafs = separate_surgical(model, surgical_first, surgical_last)
        else:
            leafs, no_leafs = separate_grad_arguments(
                model,
                grad_linear_weights,
                grad_linear_bias,
                grad_conv_weights,
                grad_conv_bias,
                grad_norm_weights,
                grad_norm_bias,
                grad_embed_weights,
            )
        leafs = ([x] if grad_input else []) + leafs
        no_leafs = ([y] if grad_input else [x, y]) + no_leafs
        # targets will never require grad

        # make leaves differentiable, turn off non-leafs
        for leaf in leafs:
            leaf.requires_grad_(True)
        for no_leaf in no_leafs:
            no_leaf.requires_grad_(False)

        if synchronize:
            maybe_synchronize(self.dev)

        return model, loss_fn, x, y, targets, leafs, no_leafs


class RuntimeMeasurement(Measurement):
    """A class to perform run time measurements of forward+backward pass."""

    def forward_backward(self, **case_kwargs) -> float:
        """Perform a forward and backward pass and return the run time.

        Syncs CUDA threads if the device is a GPU.
        Note: We directly pass input embeddings to transformers so embed weights are never used and their
        grad will be None.

        Args:
            **case_kwargs: Strings denoting which grads to compute and which to not, check docs of `Measurement.set_up()`

        Returns:
            float: The run time in seconds.
        """
        model, loss_fn, x, y, targets, leafs, no_leafs = self.set_up(**case_kwargs)

        # obtain run time
        maybe_synchronize(self.dev)
        with Timer(logger=None) as timer:
            if targets is None:
                loss_fn(model(x), y).backward()
            else:
                loss_fn(model(x, targets)).backward()
            maybe_synchronize(self.dev)

        # clean up and run checks before returning the time
        for leaf in leafs:
            assert leaf.grad is not None
            del leaf.grad
        for no_leaf in no_leafs:
            assert not hasattr(no_leaf, "grad") or no_leaf.grad is None

        return timer.last


class MemoryMeasurement(Measurement):
    """A class to measure memory usage after a forward pass."""

    def after_forward(self, **case_kwargs) -> float:
        """Return memory usage after a forward pass.

        Note: We directly pass input embeddings to transformers so embed weights are never used and their
        grad will be None.

        Args:
            **case_kwargs: Strings denoting which grads to compute and which to not, check docs of `Measurement.set_up()`

        Returns:
            float: The memory usage in bytes.
        """
        model, loss_fn, x, y, targets, leafs, no_leafs = self.set_up(**case_kwargs)

        if str(self.dev) == "cpu":

            def forward(sleep_after: float = 0.1):
                """Compute the forward pass.

                Args:
                    sleep_after: Sleep for this many seconds after the forward pass.
                        For measuring purposes Default: `0.1`.
                """
                # print(f"{loss_fn=}, {x.shape=}, {y.shape=}")
                if targets is None:
                    loss = loss_fn(model(x), y)
                else:
                    loss = loss_fn(model(x, targets)).backward()
                sleep(sleep_after)
                del loss
                gc.collect()

            gc.collect()
            return memory_usage(forward, interval=1e-3, max_usage=True)
        else:
            # @profile
            torch.cuda.reset_peak_memory_stats()

            def forward(sleep_after: float = 0.1):
                """Compute the forward pass.

                Args:
                    sleep_after: Sleep for this many seconds after the forward pass.
                        For measuring purposes Default: `0.1`.
                """
                # print(f"CUDA: {loss_fn=}, {x.shape=}, {y.shape=}")
                if targets is None:
                    loss = loss_fn(model(x), y)  # noqa: F841 (assigned to but never used)
                else:
                    loss = loss_fn(model(x, targets)).backward()  # noqa: F841 (assigned to but never used)
                # sleep(sleep_after)
                # del loss
                # gc.collect()

            # gc.collect()
            # return memory_usage(forward, interval=1e-3, max_usage=True)
            # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
            #     with record_function('forward_call'):
            forward()
            # import ipdb; ipdb.set_trace()
            return torch.cuda.max_memory_allocated() / 1024 / 1024


def separate_grad_arguments(
    model: Module,
    grad_linear_weights: bool,
    grad_linear_bias: bool,
    grad_conv_weights: bool,
    grad_conv_bias: bool,
    grad_norm_weights: bool,
    grad_norm_bias: bool,
    grad_embed_weights: bool,
) -> Tuple[List[Parameter], List[Parameter]]:
    """Separate the parameters of a model into leafs and non-leafs.

    Args:
        model: The model to separate the parameters of.
        grad_linear_weights: Whether to compute the gradient of the linear layer
            weights.
        grad_linear_bias: Whether to compute the gradient of the linear layer bias.
        grad_conv_weights: Whether to compute the gradient of the convolution layer
            weights.
        grad_conv_bias: Whether to compute the gradient of the convolution layer bias.
        grad_norm_weights: Whether to compute the gradient of the normalization layer
            weights.
        grad_norm_bias: Whether to compute the gradient of the normalization layer bias.
        grad_embed_weights: Whether to compute the gradient of the embedding layer weights

    Returns:
        A tuple of lists of parameters. The first list contains the leafs, the second
        list contains the non-leafs.

    Raises:
        NotImplementedError: If an unknown layer with parameters is encountered.
    """
    linear = (Linear, MemSaveLinear, Conv1D)
    conv = (
        Conv1d,
        Conv2d,
        Conv3d,
        ConvTranspose1d,
        ConvTranspose2d,
        ConvTranspose3d,
        MemSaveConv2d,
    )
    norm = (
        BatchNorm1d,
        BatchNorm2d,
        BatchNorm3d,
        LayerNorm,
        LayerNorm2d,
        MemSaveBatchNorm2d,
    embed = Embedding

    leafs, no_leafs = [], []

    def separate_layer(layer: Module, grad_weight: bool, grad_bias: bool):
        """Add parameters of layer to leafs or non-leafs.

        Args:
            layer: The layer whose parameters to add to (non-)leafs.
            grad_weight: Whether to compute the gradient of the layer weights.
            grad_bias: Whether to compute the gradient of the layer bias.
        """
        leafs.append(layer.weight) if grad_weight else no_leafs.append(layer.weight)
        if "bias" in layer._parameters and layer.bias is not None:
            leafs.append(layer.bias) if grad_bias else no_leafs.append(layer.bias)

    def check_lm_head(n) -> bool:
        """Checks if the module name n corresponds to an LM Head

        LM Head for transformers is not trainable (i.e. it is a module but it's weight is not a parameter)
        and the weights are tied to the embedding layer

        Args:
            n (str): name of the module

        Returns:
            bool: Whether n is a LM head
        """
        lm_head_name = getattr(model, "lm_head_name", None)
        return lm_head_name is not None and lm_head_name in n

    layers = [
        m
        for n, m in model.named_modules()
        if len(list(m.modules())) == 1 and not check_lm_head(n)
    ]

    for layer in layers:
        if isinstance(layer, linear):
            separate_layer(layer, grad_linear_weights, grad_linear_bias)
        elif isinstance(layer, conv):
            separate_layer(layer, grad_conv_weights, grad_conv_bias)
        elif isinstance(layer, norm):
            separate_layer(layer, grad_norm_weights, grad_norm_bias)
        elif isinstance(layer, embed):
            separate_layer(layer, grad_embed_weights, False)
        elif list(layer.parameters()):
            raise NotImplementedError(f"Unknown layer with parameters: {layer}.")

    return leafs, no_leafs


def separate_surgical(
    model: Module, surgical_first: bool, surgical_last: bool
) -> Tuple[List[Parameter], List[Parameter]]:
    """Separate the parameters of a model into leafs and no-leafs for surgical fine-tuning

    One and only one of `surgical_first` and `surgical_last` must be True
    Args:
        model (Module): The model to separate the parameters of.
        surgical_first (bool): Whether to compute the gradients of the first quarter of layers with parameters.
        surgical_last (bool): Whether to compute the gradients of the last quarter of layers with parameters.
    """
    assert (
        surgical_first ^ surgical_last
    ), "One and only one of surgical_first and surgical_last must be True"
    leafs, no_leafs = [], []
    counted_modules, total_modules = 0, 0

    layers = [
        m
        for n, m in model.named_modules()
        if len(list(m.modules())) == 1 and list(m.parameters())
    ]

    total_modules = len(layers)

    def separate_layer(layer: Module, leaf: bool):
        """Add parameters of layer to leafs or non-leafs.

        Args:
            layer: The layer whose parameters to add to (non-)leafs.
            leaf: Whether the layer is a leaf or not.
        """
        leafs.append(layer.weight) if leaf else no_leafs.append(layer.weight)
        if "bias" in layer._parameters and layer.bias is not None:
            leafs.append(layer.bias) if leaf else no_leafs.append(layer.bias)

    def check_condition(counted, total):
        if surgical_first:
            return counted <= total / 4
        return counted >= 3 * total / 4

    if surgical_last:
        layers = layers[::-1]
    # import ipdb; ipdb.set_trace()
    for c in layers:
        if not list(c.parameters()):
            continue
        if counted_modules <= total_modules / 4:
            # Leaf
            separate_layer(c, True)
            counted_modules += 1
        else:
            # No Leaf
            separate_layer(c, False)
            # counted_modules += surgical_last

    return leafs, no_leafs
