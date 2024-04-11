"""Measure forward pass peak memory and save to file."""

from argparse import ArgumentParser
from collections import OrderedDict
from os import makedirs, path

from memory_profiler import memory_usage
from torch import manual_seed, rand
from torch.nn import Conv2d, Sequential

from memsave_torch.nn import MemSaveConv2d

HERE = path.abspath(__file__)
HEREDIR = path.dirname(HERE)
DATADIR = path.join(HEREDIR, "raw")
makedirs(DATADIR, exist_ok=True)


parser = ArgumentParser(description="Parse arguments.")
parser.add_argument("--num_layers", type=int, help="Number of layers.")
parser.add_argument(
    "--requires_grad",
    type=str,
    choices=["all", "none", "4", "4+"],
    help="Which layers are differentiable.",
)
parser.add_argument(
    "--implementation",
    type=str,
    choices=["torch", "ours"],
    help="Which implementation to use.",
)
args = parser.parse_args()


def main():  # noqa: C901
    """Runs exps for generating the data of the visual abstract"""
    manual_seed(0)

    # create the input
    num_channels = 8
    spatial_size = 256
    batch_size = 256
    X = rand(batch_size, num_channels, spatial_size, spatial_size)

    # create the network
    # preserve input size of convolutions
    kernel_size = 3
    padding = 1

    num_layers = args.num_layers
    layers = OrderedDict()
    for i in range(num_layers):
        if args.implementation == "torch":
            layers[f"conv{i}"] = Conv2d(
                num_channels, num_channels, kernel_size, padding=padding, bias=False
            )
        elif args.implementation == "ours":
            layers[f"conv{i}"] = MemSaveConv2d(
                num_channels, num_channels, kernel_size, padding=padding, bias=False
            )
        else:
            raise ValueError(f"Invalid implementation: {args.implementation}.")

    net = Sequential(layers)

    # set differentiability
    if args.requires_grad == "none":
        for param in net.parameters():
            param.requires_grad_(False)
    elif args.requires_grad == "all":
        for param in net.parameters():
            param.requires_grad_(True)
    elif args.requires_grad == "4":
        for name, param in net.named_parameters():
            param.requires_grad_("conv3" in name)
    elif args.requires_grad == "4+":
        for name, param in net.named_parameters():
            number = int(name.replace("conv", "").replace(".weight", ""))
            param.requires_grad_(number >= 3)
    else:
        raise ValueError(f"Invalid requires_grad: {args.requires_grad}.")

    # turn off gradients for the first layer
    # net.conv0.weight.requires_grad_(False)

    # turn of gradients for all layers
    # for param in net.parameters():
    #     param.requires_grad_(False)

    # turn off all gradients except for the first layer
    # for name, param in net.named_parameters():
    #     param.requires_grad_("conv0" in name)

    # turn off all gradients except for the second layer
    # for name, param in net.named_parameters():
    #     param.requires_grad_("conv1" in name)

    # turn off all gradients except for the third layer
    # for name, param in net.named_parameters():
    #     param.requires_grad_("conv2" in name)

    for name, param in net.named_parameters():
        print(f"{name} requires_grad = {param.requires_grad}")

    # forward pass
    output = net(X)
    assert output.shape == X.shape

    return output


if __name__ == "__main__":
    max_usage = memory_usage(main, interval=1e-3, max_usage=True)
    print(f"Peak mem: {max_usage}.")
    filename = path.join(
        DATADIR,
        f"peakmem_implementation_{args.implementation}_num_layers_{args.num_layers}_requires_grad_{args.requires_grad}.txt",
    )

    with open(filename, "w") as f:
        f.write(f"{max_usage}")
