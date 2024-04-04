Use input of shape `(256, 8, 256, 256)` and size-preserving convolutions with `padding=1`, `kernel_size=3`.

---

Peak memory used by the forward pass:

- 1 layer: 1725.78515625
- 2 layers: 2238.59375
- 3 layers: 2750.390625
- 4 layers: 3261.08984375
- 5 layers: 3774.68359375

Roughly 500 MiB increase per layer added, consistent with the 512 MiB required to store an intermediate.

---

Let's turn off `requires_grad` for the first layer:

- 1 layer: 1724.75390625
- 2 layers: 2237.5703125
- 3 layers: 2749.796875
- 4 layers: 3262.453125
- 5 layers: 3773.8203125

Basically no change of effect at all!

---

Let's turn off all `requires_grad`:

- 1 layer: 1724.5390625
- 2 layers: 2238.08203125
- 3 layers: 2238.49609375
- 4 layers: 2237.92578125
- 5 layers: 2238.30078125

Now we can see that the original input, as well as two intermediates are stored at a time.

---

Let's turn off all `requires_grad` except for the first layer:

- 1 layer: 1725.52734375
- 2 layers: 2236.26953125
- 3 layers: 2749.359375
- 4 layers: 3262.171875
- 5 layers: 3773.9921875

Although we only want gradients for the first layer, we get the same memory consumption as if we wanted to compute gradients for all layers.

---

Let's turn off all `requires_grad` except for the second layer:

- 1 layer: 1725.0078125
- 2 layers: 2238.3515625
- 3 layers: 2750.6484375
- 4 layers: 3262.36328125
- 5 layers: 3774.34765625

Same behavior because we store in- and output of a convolution at a time

---

Let's turn off all `requires_grad` except for the third layer:

- 1 layer: 1725.171875
- 2 layers: 2237.85546875
- 3 layers: 2238.42578125
- 4 layers: 2749.625
- 5 layers: 3261.44921875

Notice the zero increase between 2-3 layers.

---
