"""Measure forward pass peak memory and save to file."""

from argparse import ArgumentParser
from collections import OrderedDict
from functools import partial
from os import makedirs, path

from memory_profiler import memory_usage
from torch import allclose, compile, manual_seed, rand, rand_like
from torch.autograd import grad
from torch.nn import (
    BatchNorm2d,
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
    Linear,
    Sequential,
)

from memsave_torch.nn import (
    MemSaveBatchNorm2d,
    MemSaveConv1d,
    MemSaveConv2d,
    MemSaveConv3d,
    MemSaveConvTranspose1d,
    MemSaveConvTranspose2d,
    MemSaveConvTranspose3d,
    MemSaveLinear,
)


HEREDIR = path.dirname(path.abspath(__file__))
DATADIR = path.join(HEREDIR, "raw")
makedirs(DATADIR, exist_ok=True)


def main(  # noqa: C901
    architecture: str,
    implementation: str,
    mode: str,
    num_layers: int,
    requires_grad: str,
    use_compile: bool,
):
    """Runs exps for generating the data of the visual abstract"""
    manual_seed(0)

    # create the input
    if architecture == "linear":
        X = rand(512, 1024, 256)
    elif architecture in {"conv1d", "conv_transpose1d"}:
        X = rand(4096, 8, 4096)
    elif architecture in {"conv2d", "bn2d", "conv_transpose2d"}:
        X = rand(256, 8, 256, 256)
    elif architecture in {"conv3d", "conv_transpose3d"}:
        X = rand(64, 8, 64, 64, 64)
    else:
        raise ValueError(f"Invalid argument for architecture: {architecture}.")
    assert X.numel() == 2**27  # (requires 512 MiB of storage)

    # create the network
    layers = OrderedDict()
    for i in range(num_layers):
        if architecture == "linear":
            layer_cls = {"ours": MemSaveLinear, "torch": Linear}[implementation]
            layers[f"{architecture}{i}"] = layer_cls(256, 256, bias=False)
        elif architecture == "conv1d":
            layer_cls = {"ours": MemSaveConv1d, "torch": Conv1d}[implementation]
            layers[f"{architecture}{i}"] = layer_cls(8, 8, 3, padding=1, bias=False)
        elif architecture == "conv2d":
            layer_cls = {"ours": MemSaveConv2d, "torch": Conv2d}[implementation]
            layers[f"{architecture}{i}"] = layer_cls(8, 8, 3, padding=1, bias=False)
        elif architecture == "conv3d":
            layer_cls = {"ours": MemSaveConv3d, "torch": Conv3d}[implementation]
            layers[f"{architecture}{i}"] = layer_cls(8, 8, 3, padding=1, bias=False)
        elif architecture == "bn2d":
            layer_cls = {"ours": MemSaveBatchNorm2d, "torch": BatchNorm2d}[
                implementation
            ]
            layers[f"{architecture}{i}"] = layer_cls(8)
        elif architecture == "conv_transpose1d":
            layer_cls = {"ours": MemSaveConvTranspose1d, "torch": ConvTranspose1d}[
                implementation
            ]
            layers[f"{architecture}{i}"] = layer_cls(8, 8, 3, padding=1, bias=False)
        elif architecture == "conv_transpose2d":
            layer_cls = {"ours": MemSaveConvTranspose2d, "torch": ConvTranspose2d}[
                implementation
            ]
            layers[f"{architecture}{i}"] = layer_cls(8, 8, 3, padding=1, bias=False)
        elif architecture == "conv_transpose3d":
            layer_cls = {"ours": MemSaveConvTranspose3d, "torch": ConvTranspose3d}[
                implementation
            ]
            layers[f"{architecture}{i}"] = layer_cls(8, 8, 3, padding=1, bias=False)
        else:
            raise ValueError(f"Invalid argument for architecture: {architecture}.")

    net = Sequential(layers)

    # randomly initialize parameters
    for param in net.parameters():
        param.data = rand_like(param)

    # randomly initialize running mean and std of BN
    for module in net.modules():
        if isinstance(module, (BatchNorm2d, MemSaveBatchNorm2d)):
            module.running_mean = rand_like(module.running_mean)
            module.running_var = rand_like(module.running_var)

    # set differentiability
    if requires_grad == "none":
        for param in net.parameters():
            param.requires_grad_(False)
    elif requires_grad == "all":
        for param in net.parameters():
            param.requires_grad_(True)
    elif requires_grad == "4":
        for name, param in net.named_parameters():
            param.requires_grad_(f"{architecture}3" in name)
    elif requires_grad == "4+":
        for name, param in net.named_parameters():
            number = int(name.replace(architecture, "").split(".")[0])
            param.requires_grad_(number >= 3)
    else:
        raise ValueError(f"Invalid requires_grad: {requires_grad}.")

    for name, param in net.named_parameters():
        print(f"{name} requires_grad = {param.requires_grad}")

    # set mode
    if mode == "eval":
        net.eval()
    elif mode == "train":
        net.train()
    else:
        raise ValueError(f"Invalid mode: {mode}.")

    # maybe compile
    if use_compile:
        print("Compiling model")
        net = compile(net)

    # forward pass
    output = net(X)
    assert output.shape == X.shape

    return output, net


def check_equality(
    architecture: str, mode: str, num_layers: int, requires_grad: str, use_compile: bool
):
    """Compare forward pass and gradients of PyTorch and Memsave."""
    output_ours, net_ours = main(
        architecture, "ours", mode, num_layers, requires_grad, use_compile
    )
    grad_args_ours = [p for p in net_ours.parameters() if p.requires_grad]
    grad_ours = grad(output_ours.sum(), grad_args_ours) if grad_args_ours else []

    output_torch, net_torch = main(
        architecture, "torch", mode, num_layers, requires_grad, use_compile
    )
    grad_args_torch = [p for p in net_torch.parameters() if p.requires_grad]
    grad_torch = grad(output_torch.sum(), grad_args_torch) if grad_args_torch else []

    assert allclose(output_ours, output_torch)
    assert len(list(net_ours.parameters())) == len(list(net_torch.parameters()))
    for p1, p2 in zip(net_ours.parameters(), net_torch.parameters()):
        assert allclose(p1, p2)
    assert len(grad_ours) == len(grad_torch)
    for g1, g2 in zip(grad_ours, grad_torch):
        assert allclose(g1, g2)


if __name__ == "__main__":
    # arguments
    parser = ArgumentParser(description="Parse arguments.")
    parser.add_argument("--num_layers", type=int, help="Number of layers.")
    parser.add_argument(
        "--requires_grad",
        type=str,
        choices={"all", "none", "4", "4+"},
        help="Which layers are differentiable.",
    )
    parser.add_argument(
        "--implementation",
        type=str,
        choices={"torch", "ours"},
        help="Which implementation to use.",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        choices={
            "linear",
            "conv1d",
            "conv2d",
            "conv3d",
            "bn2d",
            "conv_transpose1d",
            "conv_transpose2d",
            "conv_transpose3d",
        },
        help="Which architecture to use.",
    )
    parser.add_argument(
        "--mode", type=str, help="Mode of the network.", choices={"train", "eval"}
    )
    parser.add_argument(
        "--skip_existing", action="store_true", help="Skip existing files."
    )
    parser.add_argument(
        "--use_compile",
        action="store_true",
        help="Compile the model before the forward pass.",
    )
    args = parser.parse_args()

    filename = path.join(
        DATADIR,
        f"peakmem_{args.architecture}_mode_{args.mode}_implementation_"
        + f"{args.implementation}_num_layers_{args.num_layers}"
        + f"_requires_grad_{args.requires_grad}"
        f"{'_use_compile' if args.use_compile else ''}.txt",
    )
    if path.exists(filename) and args.skip_existing:
        print(f"Skipping existing file: {filename}.")
    else:
        # measure memory
        f = partial(
            main,
            num_layers=args.num_layers,
            requires_grad=args.requires_grad,
            implementation=args.implementation,
            architecture=args.architecture,
            mode=args.mode,
            # Memsave does not compile (TODO debug why)
            use_compile=False if args.implementation == "ours" else args.use_compile,
        )
        max_usage = memory_usage(f, interval=1e-4, max_usage=True)
        print(f"Peak mem: {max_usage}.")

        with open(filename, "w") as f:
            f.write(f"{max_usage}")

        # Memsave is not compile-able (TODO debug why)
        if not args.use_compile:
            print("Performing equality check.")
            check_equality(
                args.architecture,
                args.mode,
                args.num_layers,
                args.requires_grad,
                args.use_compile,
            )
            print("Equality check passed.")
