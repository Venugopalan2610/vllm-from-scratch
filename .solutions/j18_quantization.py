"""Reference solution, stage 18 (jax) - weight-only quantization."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

INT8_MAX = 127.0
FP8_E4M3_MAX = 448.0
TINY = 1e-8


def quantize_int8_per_channel(weight):
    """Symmetric int8, one scale for each output channel.

    The reduction axis is the LAST one. So this works on one (out, in)
    matrix and on the (layers, out, in) stacks that jvllm holds.
    """
    scales = jnp.maximum(jnp.max(jnp.abs(weight), axis=-1), TINY) / INT8_MAX
    int8_weight = jnp.clip(jnp.round(weight / scales[..., None]),
                           -INT8_MAX, INT8_MAX).astype(jnp.int8)
    return int8_weight, scales.astype(jnp.float32)


def dequantize_int8(int8_weight, scales):
    return int8_weight.astype(jnp.float32) * scales[..., None]


def quantized_matmul(inputs, int8_weight, scales):
    """inputs @ dequantize(int8_weight, scales).T, and the dequantized weight
    is NEVER made.

    A scale for each output channel commutes with the matmul. A scale on
    output row j of the weight is a scale on column j of the result. So the
    scale moves after the matmul, and decode reads the large weight only in
    its small dtype. If you make the bf16 weight first, you add a full-size write
    and a full-size read, and decode becomes slower, not faster.
    """
    return (inputs @ int8_weight.astype(inputs.dtype).T) \
        * scales.astype(inputs.dtype)


def quantize_fp8(tensor):
    """FP8 (e4m3) with one scale for the tensor. Ada (sm_89) has the dtype in
    hardware."""
    amax = jnp.maximum(jnp.max(jnp.abs(tensor)).astype(jnp.float32), TINY)
    scale = amax / FP8_E4M3_MAX
    fp8_tensor = jnp.clip(tensor.astype(jnp.float32) / scale,
                          -FP8_E4M3_MAX, FP8_E4M3_MAX)
    return fp8_tensor.astype(jnp.float8_e4m3fn), scale


def dequantize_fp8(fp8_tensor, scale, dtype=jnp.bfloat16):
    return (fp8_tensor.astype(jnp.float32) * scale).astype(dtype)


def is_quantized(leaf):
    return isinstance(leaf, dict) and "int8_weight" in leaf and "scales" in leaf


def path_name(path):
    return "/".join(str(getattr(part, "key", getattr(part, "idx", part)))
                    for part in path)


def quantize_tree(params, skip=("embed", "lm_head", "norm")):
    """int8 for each weight of 2 or more dimensions in the pytree. It is a
    tree_map, not a walk over named_modules: here the model is data.

    Skip a norm BY NAME, not by rank. jvllm stacks the layers, so a norm is
    (num_layers, hidden). That is 2-D, and a rule `ndim >= 2` quantizes it.
    A norm is one vector for each layer: it saves nothing, and the model is
    most sensitive there.
    """
    def quantize_leaf(path, leaf):
        if jnp.ndim(leaf) < 2 or any(part in path_name(path) for part in skip):
            return leaf
        int8_weight, scales = quantize_int8_per_channel(leaf)
        return {"int8_weight": int8_weight, "scales": scales}

    return jax.tree_util.tree_map_with_path(quantize_leaf, params)


def dequantize_tree(quantized_tree, dtype=jnp.bfloat16):
    def dequantize_leaf(leaf):
        if not is_quantized(leaf):
            return leaf
        return dequantize_int8(leaf["int8_weight"],
                               leaf["scales"]).astype(dtype)

    return jax.tree.map(dequantize_leaf, quantized_tree, is_leaf=is_quantized)


def array_bytes(array):
    return array.size * array.dtype.itemsize


def tree_bytes(tree):
    return int(sum(array_bytes(array) for array in jax.tree.leaves(tree)))


def continuation_logits(model, token_ids, prompt_len):
    tokens = jnp.asarray([token_ids], dtype=jnp.int32)
    logits, _ = model.forward(tokens, logits_index=None)        # (1, T, V)
    return logits[0, prompt_len - 1:-1].astype(jnp.float32)


@dataclass
class Fidelity:
    top1_agreement: float       # the fraction of positions with the same top token
    mean_kl: float              # the mean KL(reference || candidate), in nats


def fidelity(reference_logits, candidate_logits):
    reference = jax.nn.log_softmax(reference_logits.astype(jnp.float32), axis=-1)
    candidate = jax.nn.log_softmax(candidate_logits.astype(jnp.float32), axis=-1)
    same_choice = jnp.argmax(reference, axis=-1) == jnp.argmax(candidate, axis=-1)
    kl = (jnp.exp(reference) * (reference - candidate)).sum(axis=-1)
    return Fidelity(float(same_choice.mean()), float(kl.mean()))
