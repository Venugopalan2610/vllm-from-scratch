"""Stage 18 (JAX) - weight-only quantization over a pytree.

`./vc lore 18 --jax` for the insight. `./vc test 18 --jax` to check yourself.

Decode reads every weight to produce one token, so halving the weight bytes
nearly halves decode time. That is the whole idea, and it is the same
arithmetic on both tracks.

The shape of the model is different. torch has a graph of Modules. You walk it
with named_modules(), and you put a quantized layer in the place of each
nn.Linear.

Here the model is a PYTREE: nested dicts of arrays, with no objects and no
forward methods. So you quantize it with a tree_map. You replace nothing. You
build a new tree.

WHAT YOU ARE BUILDING

    quantize_int8_per_channel(weight) -> (int8_weight, scales fp32)
    dequantize_int8(int8_weight, scales) -> fp32
    quantized_matmul(inputs, int8_weight, scales)
        -> inputs @ dequantize(int8_weight, scales).T
    quantize_fp8(tensor) -> (fp8_tensor float8_e4m3fn, scale)
    dequantize_fp8(fp8_tensor, scale, dtype=bfloat16)

    quantize_tree(params, skip=("embed", "lm_head", "norm")) -> quantized pytree
    dequantize_tree(quantized_tree, dtype=bfloat16) -> params pytree
    tree_bytes(tree) -> int
    continuation_logits(model, token_ids, prompt_len) -> (OSL, vocab) fp32
    fidelity(reference_logits, candidate_logits) -> Fidelity

A quantized leaf is `{"int8_weight": ..., "scales": ...}` in place of the
array. That
keeps the tree a plain nested dict, so `jax.tree.map(..., is_leaf=...)` still
works on it and you do not need a custom pytree node.

THE ONE THAT MATTERS: KEEP THE DEQUANT IN THE EPILOGUE

A per-output-channel scale COMMUTES with the matmul. To scale every element of
output row j by s_j is the same as to scale column j of the result:

    x @ (q * s[:, None]).T   ==   (x @ q.T) * s

The left side builds a full-size dequantized weight. That is a write and a
read of exactly the tensor that you tried not to read. The right side never
builds it. Get this backwards, and int8 makes decode SLOWER than bf16. That is
a very confusing bug.

    quantized_matmul(x, q, scales) = (x @ q.astype(x.dtype).T) * scales

Here x is the inputs, q is int8_weight and s is scales.

SHAPES

jvllm stacks the weights of the layers, so `params["layers"]["q_proj"]` is
(num_layers, out, in), and not (out, in). Make `quantize_int8_per_channel`
reduce over the LAST axis. It then works on both shapes with no special case.
The scales come out as (num_layers, out), and `scales[..., None]` broadcasts
correctly.

WHAT NOT TO QUANTIZE

  - The norms. Skip them BY NAME, and not by rank. jvllm stacks the layers,
    so a norm arrives as (num_layers, hidden), which has two dimensions. So a
    rule "ndim >= 2 means a matrix" quantizes every norm in the model. A norm
    is one vector for each layer. It saves nothing, and the model is most
    sensitive there.
  - The embedding and lm_head, by default. They are a big fraction of a 0.6B
    model's parameters, and they are also where int8 hurts the accuracy most.
    Skip them by name, and notice how much of the "halved the model" claim
    quietly depends on that choice.

FP8

`jnp.float8_e4m3fn` is a real dtype. Max representable is 448, so the scale is
amax / 448, and you clip to that range. Use one scale for each tensor, and not
for each channel. fp8 has enough exponent range, so a scale for each channel
buys much less than it does for int8.
"""

from dataclasses import dataclass


def quantize_int8_per_channel(weight):
    """Symmetric int8, with one scale for each output channel."""
    raise NotImplementedError("stage 18 (jax): implement quantize_int8_per_channel")


def dequantize_int8(int8_weight, scales):
    raise NotImplementedError("stage 18 (jax): implement dequantize_int8")


def quantized_matmul(inputs, int8_weight, scales):
    """inputs @ dequantize(int8_weight, scales).T. Do not make the dequantized
    weight."""
    raise NotImplementedError("stage 18 (jax): implement quantized_matmul")


def quantize_fp8(tensor):
    """e4m3, with one scale for the tensor."""
    raise NotImplementedError("stage 18 (jax): implement quantize_fp8")


def dequantize_fp8(fp8_tensor, scale, dtype=None):
    raise NotImplementedError("stage 18 (jax): implement dequantize_fp8")


def quantize_tree(params, skip=("embed", "lm_head", "norm")):
    """int8 for each weight of 2 or more dimensions, with a tree_map."""
    raise NotImplementedError("stage 18 (jax): implement quantize_tree")


def dequantize_tree(quantized_tree, dtype=None):
    raise NotImplementedError("stage 18 (jax): implement dequantize_tree")


def tree_bytes(tree):
    """The bytes of a params tree. Count a quantized leaf correctly."""
    raise NotImplementedError("stage 18 (jax): implement tree_bytes")


def continuation_logits(model, token_ids, prompt_len):
    """The float32 logits that predict each output token. -> (OSL, vocab).

    token_ids is a prompt of prompt_len tokens, then its continuation of OSL
    tokens. Row i predicts token_ids[prompt_len + i], so it comes from the
    position before that token. One forward pass with logits_index=None gives
    the logits of every position.
    """
    raise NotImplementedError("stage 18 (jax): implement continuation_logits")


@dataclass
class Fidelity:
    top1_agreement: float       # the fraction of positions with the same top token
    mean_kl: float              # the mean KL(reference || candidate), in nats


def fidelity(reference_logits, candidate_logits):
    """How close the candidate model stays to the reference, position by
    position. Both arguments are (positions, vocab). -> Fidelity.

    top1_agreement: the fraction of positions where the two models choose the
        same most likely token.
    mean_kl: the mean of KL(P || Q), with P from the reference and Q from the
        candidate. Compute it from log_softmax in float32.

    Both models read the SAME tokens, so the difficulty of the text and its
    length cancel. That is why this metric is stable, and perplexity is not.
    """
    raise NotImplementedError("stage 18 (jax): implement fidelity")
