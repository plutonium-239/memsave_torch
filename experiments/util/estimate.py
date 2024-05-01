"""Estimate possible speed-up when randomizing the weight VJP of convolutions.

We take a CNN and answer the following questions:

Q1) What is the relative run time consumed by the weight VJP for convolutions?

Q2) Assume we achieve a speed-up ``x`` by randomizing the weight VJP, what would
    be the speed-up for one optimization step (forward+backward)?

Q3) The same as Q1) and Q2) but in terms of memory consumption.
"""

import argparse
import os
from typing import Callable, Dict, List, Optional

from torch import Tensor, device, manual_seed, rand, randint
from torch.nn import CrossEntropyLoss, Module

from experiments.util import models
from experiments.util.measurements import (
    MemoryMeasurement,
    RuntimeMeasurement,
)

allowed_cases = [
    "grad_linear_weights",
    "no_grad_linear_weights",
    "grad_linear_bias",
    "no_grad_linear_bias",
    "grad_conv_weights",
    "no_grad_conv_weights",
    "grad_conv_bias",
    "no_grad_conv_bias",
    "grad_norm_weights",
    "no_grad_norm_weights",
    "grad_norm_bias",
    "no_grad_norm_bias",
    "grad_input",
    "no_grad_input",
    "grad_embed_weights",
    "no_grad_embed_weights",
]


def parse_case(case: Optional[List[str]]) -> Dict[str, bool]:
    """Small helper function to convert cases into kw-arguments for measurements

    Args:
        case (Optional[List[str]]): List of all cases

    Returns:
        Dict[str, bool]: dictionary with keys as allowed_cases present in the input (which dont start with ``no_``)
    """
    kw = {}
    if case is None:
        return kw
    for c in case:
        if c.startswith("no_"):
            kw[c[3:]] = False
        else:
            kw[c] = True
    return kw


def skip_case_check(args: argparse.Namespace) -> bool:
    """Decide whether to skip the case:

    1. when case has grad_norm_* but model does not have any normalization layers
    2. when case has no_grad_embed_weights but no grad_input: there is a backward error (no input requires_grad)

    Args:
        args (argparse.Namespace): args

    Returns:
        bool: Whether to skip or not
    """
    invalid = False
    if args.case is None:
        return invalid
    # 1.
    for c in ["grad_norm_bias", "grad_norm_weights"]:
        if c in args.case and args.model in models.models_without_norm:
            invalid = True
    for c in ["no_grad_norm_bias", "no_grad_norm_weights"]:
        if c not in args.case and args.model in models.models_without_norm:
            invalid = True
    # 2.
    if "no_grad_embed_weights" in args.case and "grad_input" not in args.case:
        invalid = True
    if invalid:
        if args.print:
            print("-1")
            return invalid
        # else:
        with open(
            os.path.join(
                args.results_dir, f"raw/{args.estimate}-{args.architecture}.txt"
            ),
            "a",
        ) as f:
            f.write("-1\n")
    return invalid


def estimate_speedup(
    model_fn: Callable[[], Module],
    loss_fn: Callable[[], Module],
    x: Tensor,
    y: Tensor,
    targets: Optional[List[Dict[str, Tensor]]],
    architecture: str,
    dev: device,
    case: List[str],
    results_dir: str,
    return_val: bool = False,
):
    """Save an estimate of total training speed-up caused by a weight VJP speed-up.

    Args:
        model_fn: Function that sets up the neural network.
        loss_fn: Function that sets up the loss function.
        x: Input to the model.
        y: Labels of the input.
        targets: Targets in case of detection model
        architecture: linear or conv
        dev: Device to run the computation on.
        case: str indicating which grads to take
        results_dir: See args.results_dir
        return_val: Whether to return the value or save it (Default: Save)

    Returns:
        result: The required estimate (only returned if return_val is True)
    """
    # print(f"{model_fn.__name__} {loss_fn.__name__} input {tuple(x.shape)} {str(dev)}")
    # print(78 * "=")

    timer = RuntimeMeasurement(model_fn, loss_fn, x, y, dev, targets)

    # PyTorch's profiler does not track functions on the C-layer which would be
    # required to disentangle the backward through convolutions into weight- and
    # input-backprop. Instead, we compute the gradient w.r.t. (1) all model
    # parameters, (2) the input and all model parameters, and (3) the input, all
    # biases, and all Linear layer weights. Then we use (2) - (3) as a proxy for
    # the computation time of the convolution weights

    kw = parse_case(case)
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
    #     with record_function('model_run_custom'):
    result = timer.forward_backward(**kw)

    # import ipdb; ipdb.set_trace()
    # print(prof.key_averages().table(sort_by="cuda_time_total"))

    if return_val:
        return result
    with open(os.path.join(results_dir, f"raw/time-{architecture}.txt"), "a") as f:
        # f.write(f"{args.model},{loss_fn.__name__},{dev},{case},{result},{x.shape},{y.shape}\n")
        f.write(f"{result}\n")


def estimate_mem_savings(
    model_fn: Callable[[], Module],
    loss_fn: Callable[[], Module],
    x: Tensor,
    y: Tensor,
    targets: Optional[List[Dict[str, Tensor]]],
    architecture: str,
    dev: device,
    case: List[str],
    results_dir: str,
    return_val: bool = False,
):
    """Print an estimate of the memory savings caused by weight VJP memory savings.

    Args:
        model_fn: Function that sets up the neural network.
        loss_fn: Function that sets up the loss function.
        x: Input to the model.
        y: Labels of the input.
        targets: Targets in case of detection model
        architecture: linear or conv
        dev: Device to run the computation on.
        case: str indicating which grads to take
        results_dir: See args.results_dir
        return_val: Whether to return the value or save it (Default: Save)

    Returns:
        result: The required estimate (only returned if return_val is True)
    """
    # print(f"{model_fn.__name__} {loss_fn.__name__} input {tuple(x.shape)} {str(dev)}")
    # print(78 * "=")

    memory = MemoryMeasurement(model_fn, loss_fn, x, y, dev, targets)

    # We compute the memory consumption right after the forward pass when
    # computing the gradient w.r.t. (1) all model parameters, (2) the input and
    # all model parameters, and (3) the input, all biases, and all Linear layer
    # weights. Then we use (2) - (3) as a proxy for the memory used by the
    # convolution weights

    kw = parse_case(case)
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
    #     with record_function('model_run_custom'):
    result = memory.after_forward(**kw)

    # import ipdb; ipdb.set_trace()

    if return_val:
        return result
    with open(os.path.join(results_dir, f"raw/memory-{architecture}.txt"), "a") as f:
        # f.write(f"{args.model},{loss_fn.__name__},{dev},{case},{result},{x.shape},{y.shape}\n")
        f.write(f"{result}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", "-B", type=int, default=64, help="Batch Size")
    parser.add_argument(
        "--input_channels", "-C_in", type=int, default=3, help="Input Channels"
    )
    parser.add_argument(
        "--input_hw",
        "-HW",
        type=int,
        default=256,
        help="Input width and height (equal)",
    )
    parser.add_argument(
        "--num_classes", "-n_class", type=int, default=1000, help="No. of classes"
    )
    parser.add_argument("--model", type=str, required=True, help="Which model to run")
    parser.add_argument(
        "--architecture",
        type=str,
        required=True,
        help="Which architecture to run",
        choices=["conv", "linear", "transformer"],
    )
    parser.add_argument(
        "--estimate",
        type=str,
        required=True,
        choices=["time", "memory"],
        help="What to estimate (time/memory)",
    )
    parser.add_argument(
        "--case",
        type=str,
        nargs="+",
        # required=True, # allow to be None
        choices=allowed_cases,
        help=f"Which case to run, allowed values are {allowed_cases}",
    )
    parser.add_argument("--device", type=str, default="cpu", help="torch device name")
    parser.add_argument(
        "--print",
        action="store_true",
        default=False,
        help="Print result to stdout instead of writing to file",
    )
    parser.add_argument(
        "--results_dir", type=str, default="results/", help="the base results dir"
    )

    args = parser.parse_args()

    assert os.path.exists(args.results_dir)
    if not skip_case_check(args):
        dev = device(args.device)

        # ImageNet toy data
        batch_size = args.batch_size
        num_classes = args.num_classes
        models.num_classes = num_classes

        if args.architecture == "conv":
            input_shape = (args.input_channels, args.input_hw, args.input_hw)
            models.conv_input_shape = input_shape
            model_fn = models.conv_model_fns.get(args.model)
            y_args = {"size": (batch_size,), "low": 0, "high": num_classes}
            assert (
                model_fn is not None
            ), f"Conv model name {args.model} not found, must be one of {list(models.conv_model_fns.keys())}"
        elif args.architecture == "linear":
            input_shape = [args.input_hw**2]
            models.linear_input_shape = input_shape[0]
            model_fn = models.linear_model_fns.get(args.model)
            y_args = {"size": (batch_size,), "low": 0, "high": num_classes}
            assert (
                model_fn is not None
            ), f"Linear model name {args.model} not found, must be one of {list(models.linear_model_fns.keys())}"
        elif args.architecture == "transformer":
            vocab_dim = args.num_classes
            embed_dim = args.input_channels
            seq_len = args.input_hw
            model_fn = models.transformer_model_fns.get(args.model)
            if args.model in models.hf_transformers_models:
                model_fn_orig = model_fn
                model_fn = lambda: models.TransformersModelWrapper(model_fn_orig)  # noqa: E731
                config = models.get_transformers_config(args.model)
                # as per transformers.PretrainedConfig these 2 should be present in all models:
                vocab_dim = config.vocab_size
                embed_dim = config.hidden_size
            models.transformer_input_shape = (vocab_dim, embed_dim)
            input_shape = [seq_len, embed_dim]
            y_args = {"size": (batch_size, seq_len), "low": 0, "high": vocab_dim}
            assert (
                model_fn is not None
            ), f"Transformer model name {args.model} not found, must be one of {list(models.transformer_model_fns.keys())}"

        loss_fn = CrossEntropyLoss

        manual_seed(0)  # make deterministic

        x = rand(batch_size, *input_shape, device=dev)
        y = randint(**y_args, device=dev)
        targets = None
        if args.model in models.detection_models:
            # pred is a dictionary of losses
            num_boxes = 2
            batch_size = 4
            x = rand(batch_size, *input_shape, device=dev)
            boxes = rand(batch_size, num_boxes, 4)
            boxes[:, :, 2:4] = boxes[:, :, 0:2] + boxes[:, :, 2:4]
            labels = randint(num_classes // 50, (batch_size, num_boxes))
            targets = [
                {"boxes": boxes[i], "labels": labels[i]} for i in range(batch_size)
            ]
            y = Tensor([])
            loss_fn = models.DetectionLossWrapper
        elif args.model in models.segmentation_models:
            model_fn_orig = model_fn
            model_fn = lambda: model_fn_orig(num_classes=num_classes // 50)  # noqa: E731
            y = randint(
                size=(batch_size, args.input_hw, args.input_hw),
                low=0,
                high=num_classes // 50,
                device=dev,
            )
            loss_fn_orig = loss_fn
            loss_fn = lambda: models.SegmentationLossWrapper(loss_fn_orig)  # noqa: E731

        # warm-up
        # with redirect_stdout(open(devnull, "w")):
        #     estimate_speedup(model_fn, loss_fn, x, y, dev, vjp_speedups[:1])
        # print('initial memory:', cuda.max_memory_allocated()/1024/1024)

        if args.estimate == "time":
            res = estimate_speedup(
                model_fn,
                loss_fn,
                x,
                y,
                targets,
                args.architecture,
                dev,
                args.case,
                args.results_dir,
                args.print,
            )
        elif args.estimate == "memory":
            res = estimate_mem_savings(
                model_fn,
                loss_fn,
                x,
                y,
                targets,
                args.architecture,
                dev,
                args.case,
                args.results_dir,
                args.print,
            )
        if args.print:
            print(res)
