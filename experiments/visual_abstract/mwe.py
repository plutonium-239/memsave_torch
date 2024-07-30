"""PyTorch's convolutions store their input when `weight.requires_grad=False`."""

from collections import OrderedDict

from memory_profiler import memory_usage
from torch import rand
from torch.nn import Conv2d, Sequential

SHAPE_X = (256, 8, 256, 256)  # shape of the input
MEM_X = 512  # requires 512 MiB storage
NUM_LAYERS = 5


def setup():
    """Create a deep linear CNN with size-preserving convolutions and an input."""
    layers = OrderedDict()
    for i in range(NUM_LAYERS):
        layers[f"conv{i}"] = Conv2d(8, 8, 3, padding=1, bias=False)
    return rand(*SHAPE_X), Sequential(layers)


# Consider three different scenarios: 1) no parameters are trainable, 2) all
# layers are trainable, 3) only the first layer is trainable
def non_trainable():
    """Forward pass through the CNN with all layers non-trainable."""
    X, net = setup()
    for i in range(NUM_LAYERS):
        getattr(net, f"conv{i}").weight.requires_grad = False

    for name, param in net.named_parameters():
        print(f"{name}, requires_grad={param.requires_grad}")

    return net(X)


def all_trainable():
    """Forward pass through the CNN with all layers trainable."""
    X, net = setup()
    for i in range(NUM_LAYERS):
        getattr(net, f"conv{i}").weight.requires_grad = True

    for name, param in net.named_parameters():
        print(f"{name}, requires_grad={param.requires_grad}")

    return net(X)


def first_trainable():
    """Forward pass through the CNN with first layer trainable."""
    X, net = setup()
    for i in range(NUM_LAYERS):
        getattr(net, f"conv{i}").weight.requires_grad = i == 1

    for name, param in net.named_parameters():
        print(f"{name}, requires_grad={param.requires_grad}")

    return net(X)


if __name__ == "__main__":
    kwargs = {"interval": 1e-4, "max_usage": True}  # memory profiler settingss

    # measure memory and print
    mem_setup = memory_usage(setup, **kwargs)
    print(f"Weights+input: {mem_setup:.1f} MiB.")

    mem_non = memory_usage(non_trainable, **kwargs) - mem_setup
    print(
        f"Non-trainable: {mem_non:.1f} MiB (≈{mem_non / MEM_X:.1f} hidden activations)."
    )

    mem_all = memory_usage(all_trainable, **kwargs) - mem_setup
    print(
        f"All-trainable: {mem_all:.1f} MiB (≈{mem_all / MEM_X:.1f} hidden activations)."
    )

    mem_first = memory_usage(first_trainable, **kwargs) - mem_setup
    print(
        f"First-trainable: {mem_first:.1f} MiB (≈{mem_first / MEM_X:.1f} hidden activations)."
    )
