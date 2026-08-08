"""Stage 18 (JAX) - weight-only quantization over a pytree.

`./vc lore 18 --jax` for the insight. `./vc test 18 --jax` to check yourself.

Decode reads every weight to produce one token, so halving the weight bytes
nearly halves decode time. That is the whole idea, and it is the same
arithmetic on both tracks.

What is different is the shape of the model. torch has a graph of Modules and
you walk it with named_modules(), swapping nn.Linear for a quantized one. Here
the model is a PYTREE -- nested dicts of arrays, no objects, no forward methods
-- so quantizing it is a tree_map. Nothing is replaced; a new tree is built.

WHAT YOU'RE BUILDING

    quantize_int8_per_channel(W) -> (q int8, scales fp32)
    dequantize_int8(q, scales) -> fp32
    quantized_matmul(x, q, scales) -> x @ dequantize(q, scales).T
    quantize_fp8(t) -> (q float8_e4m3fn, scale)
    dequantize_fp8(q, scale, dtype=bfloat16)

    quantize_tree(params, skip=("embed", "lm_head", "norm")) -> quantized pytree
    dequantize_tree(qtree, dtype=bfloat16) -> params pytree
    tree_bytes(tree) -> int
    perplexity(model, text, max_len=512) -> float

A quantized leaf is `{"q": ..., "scales": ...}` in place of the array. That
keeps the tree a plain nested dict, so `jax.tree.map(..., is_leaf=...)` still
works on it and you do not need a custom pytree node.

THE ONE THAT MATTERS: KEEP THE DEQUANT IN THE EPILOGUE

A per-output-channel scale COMMUTES with the matmul. Scaling every element of
output row j by s_j is the same as scaling column j of the result:

    x @ (q * s[:, None]).T   ==   (x @ q.T) * s

The left side builds a full-size dequantized weight -- a write and a read of
exactly the tensor you were trying not to read. The right side never does. Get
this backwards and int8 makes decode SLOWER than bf16, which is a genuinely
confusing thing to debug.

    quantized_matmul(x, q, scales) = (x @ q.astype(x.dtype).T) * scales

SHAPES

jvllm stacks each layer's weights, so `params["layers"]["q_proj"]` is
(num_layers, out, in), not (out, in). Write `quantize_int8_per_channel` to
reduce over the LAST axis and it works on both without a special case -- scales
come out as (num_layers, out), and `scales[..., None]` broadcasts correctly.

WHAT NOT TO QUANTIZE

  - The norms -- and skip them BY NAME, not by rank. Because layers are
    stacked, a norm arrives as (num_layers, hidden), which is two-dimensional,
    so an "ndim >= 2 means it is a matrix" rule quantizes every norm in the
    model. They are one vector per layer: nothing to save, and it is where the
    model is most sensitive.
  - The embedding and lm_head, by default. They are a big fraction of a 0.6B
    model's parameters, and they are also where int8 hurts perplexity most.
    Skip them by name, and notice how much of the "halved the model" claim
    quietly depends on that choice.

FP8

`jnp.float8_e4m3fn` is a real dtype. Max representable is 448, so the scale is
amax / 448 and you clip to that range. Per tensor, not per channel -- fp8 has
enough exponent range that per-channel scales buy much less than they do for
int8.
"""


def quantize_int8_per_channel(W):
    """Symmetric int8 with one scale per output channel."""
    raise NotImplementedError("stage 18 (jax): implement quantize_int8_per_channel")


def dequantize_int8(q, scales):
    raise NotImplementedError("stage 18 (jax): implement dequantize_int8")


def quantized_matmul(x, q, scales):
    """x @ dequantize(q, scales).T without materializing the dequantized W."""
    raise NotImplementedError("stage 18 (jax): implement quantized_matmul")


def quantize_fp8(t):
    """Per-tensor e4m3."""
    raise NotImplementedError("stage 18 (jax): implement quantize_fp8")


def dequantize_fp8(q, scale, dtype=None):
    raise NotImplementedError("stage 18 (jax): implement dequantize_fp8")


def quantize_tree(params, skip=("embed", "lm_head", "norm")):
    """int8 every 2-D-or-bigger weight in the tree, by tree_map."""
    raise NotImplementedError("stage 18 (jax): implement quantize_tree")


def dequantize_tree(qtree, dtype=None):
    raise NotImplementedError("stage 18 (jax): implement dequantize_tree")


def tree_bytes(tree):
    """Total bytes of a params tree, quantized leaves counted correctly."""
    raise NotImplementedError("stage 18 (jax): implement tree_bytes")


def perplexity(model, text, max_len=512):
    """Token-level perplexity of `text` under `model`."""
    raise NotImplementedError("stage 18 (jax): implement perplexity")
