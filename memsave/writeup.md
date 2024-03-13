# Saving Memory in CNNs (with PyTorch)

CNNs are made of mainly Convolution layers along with activations such as ReLU and normalization/pooling layers such as batch normalization and max pooling.

## Motivation

In torch these layers are implemented by `torch.nn.Conv2d`, `torch.nn.ReLU`, `torch.nn.BatchNorm2d`,`torch.nn.MaxPool2d`, and give rise to very fast code that calls native C++ and CUDA kernels at the lowest level. This holds for the forward pass as well as the backward pass. However, sometimes memory efficiency is traded off for better time efficiency by storing a lot of tensors that we *might* need. But this results in consumer-level GPUs to not have enough VRAM for performing these tasks with a decent batch size. Here, we try to make that possible by implementing our own memory saving layers while also not giving up time efficiency. These can be found in the `memsave` module.

==An important use case is when you want to slightly alter a layer==

## Setup

Let's start with a ResNet-101 model. It is a good candidate as it is decently powerful and modern and is frequently used as a backbone for many other architectures such as CLIP and LAVA. Our experiments are run using the `rundemo.py` script. We add `resnet101` to the `models` list in this file. This script essentially tells us how much memory is taken by the information to be used in the kernel weights gradients/VJPs (tensors that are stored in the forward pass to be used in the backward pass). An explanation on VJPs is outside the scope of this writeup. This is done by turning `reuires_grad` off on the tensor w.r.t. we want to find the VJP information memory usage.
Since we are interested in the conv weight VJPs, information for that is essentially the input to the conv layer.
We fix the input size to `[64,3,224,224]`.
These experiments are run on a NVIDIA GTX 1660Ti mobile, which has 6GB of VRAM.

## Results and Analysis

### Base ResNet-101

Running `rundemo.py` with `resnet101`, we get:

```
=======================resnet101 input (64,3,224,224) cuda=======================
M(forward + grad params): 7995.663MB
M(forward + grad (x + params)): 7995.663MB
M(forward + grad (x + params - conv_weights)): 7995.663MB
Information for conv weight VJPs uses 0.0% of memory
```

It says that the inputs use 0% of memory. That seems incorrect, but on observing the memory usage on line 4, we can tell that even though we turned off `requires_grad` on the weights, `torch` did not take that into account and save less stuff. This is probably because in most training uses cases, you want the conv weights to be trained as well, and a lot of these tensors are saved from activations before the input anyways. More on this later.

### MemSaveConv2d

Let's try replacing all the `torch.nn.Conv2d`s in this ResNet-101 with `memsave.MemSaveConv2d`s. The `memsave` module has a handy function `convert_to_memory_saving` which takes in any `nn.Module`(in this case our ResNet-101 from `torchvision`) and gives back a model with layers converted to their memory saving types (with some user choices on which layers to replace). We can specify only `conv2d=True`[^1] to convert only the `Conv2d` layers. Our implementation saves memory by only saving inputs and weights when they would be required in the gradient calculation for weights and inputs respectively.

Again, running `rundemo.py`, this gives us:
```
=================memsave_resnet101 input (64,3,224,224) cuda==================
M(forward + grad params): 7995.663MB
M(forward + grad (x + params)): 7995.663MB
M(forward + grad (x + params - conv_weights)): 7946.663MB
Information for conv weight VJPs uses 0.6% of memory
```

Only 0.6%? That isn't really saving much. Let's investigate by looking at one of the `Bottleneck` blocks in ResNet-101 (and the size of the input tensor they receive).
```
Bottleneck: 2-1                   [7, 64, 56, 56] 
   └─Conv2d: 3-1                  [7, 64, 56, 56] 
   └─BatchNorm2d: 3-2             [7, 64, 56, 56] 
   └─ReLU: 3-3                    [7, 64, 56, 56] 
   └─Conv2d: 3-4                  [7, 64, 56, 56] 
   └─BatchNorm2d: 3-5             [7, 64, 56, 56] 
   └─ReLU: 3-6                    [7, 64, 56, 56] 
   └─Conv2d: 3-7                  [7, 64, 56, 56] 
   └─BatchNorm2d: 3-8             [7, 256, 56, 56]
   └─Sequential: 3-9              [7, 64, 56, 56] 
   │    └─Conv2d: 4-1             [7, 64, 56, 56] 
   │    └─BatchNorm2d: 4-2        [7, 256, 56, 56]
   └─ReLU: 3-10                   [7, 256, 56, 56]
```

We see that it's not made of just `Conv2d` layers; there are multiple `Relu` and `BatchNorm2d` layers in the mix as well. Let's look at `ReLU` first since it is a simple non-trainable activation. Note that it still needs to store something to propagate the gradient (specifically which inputs were > 0). In `torch.nn.ReLU`, this is done by saving the output tensor and checking which elements are > 0. This means saving the whole activation which is of the same size as the input, i.e. `[7, channels, 56, 56]` (for this example), which mostly negates any storage we gained from using `MemSaveConv2d`s. So, let's replace those too!

### MemSaveConv2d + MemSaveReLU + MemSaveBatchNorm2d

Our implementation for `MemSaveReLU` saves a boolean mask instead of the whole floating point output. This reduces our storage needs by 4x for `float32` inputs (or even 8x for `float64` inputs). Unfortunately, `torch` only supports `1-byte bool` and `1-bit bool`s are being worked on right now. That would reduce our storage needs by **32x (or even 64x)!** For `MemSaveBatchNorm2d`, we again only save weights/inputs as in `MemSaveConv2d`. 
Specifying `conv2d=True, relu=True, batchnorm2d=True` in our convert function, we run `rundemo.py` again:

```
=================memsave_resnet101 input (64,3,224,224) cuda==================
M(forward + grad params): 8923.433MB
M(forward + grad (x + params)): 8923.433MB
M(forward + grad (x + params - conv_weights)): 5536.370MB
Information for conv weight VJPs uses 38.0% of memory
```

That's great, we went from 0.6% to 38.0% savings!

### Bonus: MemSaveMaxPool2d and MemSaveLinear

We also have implementations for `MaxPool2d` and `Linear` layers, as they are frequently used in CNNs. ResNet-101 only has 1 `MaxPool2d` and `Linear` layer (there is also 1 `AdaptiveAvgPool2d`, which we are working on), so we dont expect to see a major difference. We get:
```
=================memsave_resnet101 input (64,3,224,224) cuda==================
M(forward + grad params): 8727.683MB
M(forward + grad (x + params)): 8727.683MB
M(forward + grad (x + params - conv_weights)): 5341.370MB
Information for conv weight VJPs uses 38.8% of memory
```

As expected, we gained only an extra 0.8% of memory savings.

So, using just the memory saving conv2d, we were not able to achieve much, but using it conjunction with the memory saving relu and batchnorm2d made a huge impact on the required memory - from 8.7 GB to 5.3 GB.

**Now it fits in the 1660ti's VRAM without overflowing to system memory!** This enables the GPU to not have to go through the CPU to access system memory.

## Time comparison

In this section, we compare the time taken by the forward passes for the base model and the memory saving model.

### Base

```
=====================resnet101 input (64,3,224,224) cuda======================
T(forward + grad params): 20.128s
T(forward + grad (x + params)): 20.183s
T(forward + grad (x + params - conv_weights)): 9.625s
```

When not finding the gradients for conv weights, `torch` may not be memory efficient but it certainly is time efficient and skips the calculation which results in the 11 second time difference. Let's see how the memory saving model fares.

### MemSave

```
=================memsave_resnet101 input (64,3,224,224) cuda==================
T(forward + grad params): 26.932s
T(forward + grad (x + params)): 26.867s
T(forward + grad (x + params - conv_weights)): 2.054s
```

At first, we see that the time for a forward pass increased from 20s to 26s. This can be attributed to the conditional saving of tensors. But, then we look at the third result in which we dont need to save conv weights, and the memory saving model beats out the base model **by a multiple of 4.6**! This is the reduction in time from not having to save and access tensors, some of which overflow to the system memory in the base model.

==Clearly, `MemSave` has it's use cases - if you are just training a pre-defined CNN on some task, it is probably not very beneficial for you. However, if you have a use case in which you want to explicitly not==

---

[^1]: In the `convert_to_memory_saving` function, by default all arguments `linear, conv2d, batchnorm2d, relu, maxpool2d` are `True` so we need to also specify the others to `False`.

permute layers - degrees of freedom

aot compilation

compare with torch.no_grad