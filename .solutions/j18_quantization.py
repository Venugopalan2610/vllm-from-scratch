"""Reference solution, stage 18 (jax) - weight-only quantization."""

import jax
import jax.numpy as jnp


def quantize_int8_per_channel(W):
    """Symmetric int8, one scale per output channel.

    The reduction axis is the LAST one, so this works unchanged on a single
    (out, in) matrix and on the (layers, out, in) stacks jvllm actually holds.
    """
    scales = jnp.maximum(jnp.max(jnp.abs(W), axis=-1), 1e-8) / 127.0
    q = jnp.clip(jnp.round(W / scales[..., None]), -127, 127).astype(jnp.int8)
    return q, scales.astype(jnp.float32)


def dequantize_int8(q, scales):
    return q.astype(jnp.float32) * scales[..., None]


def quantized_matmul(x, q, scales):
    """x @ dequantize(q, scales).T, WITHOUT ever building the dequantized W.

    A per-output-channel scale commutes with the matmul: scaling every element
    of output row j by s_j is the same as scaling column j of the result. So
    the scale moves into the epilogue and the big weight is only ever read in
    its small dtype. Materialise the bf16 weight first and you have added a
    full-size write and a full-size read -- and made decode slower, not faster.
    """
    return (x @ q.astype(x.dtype).T) * scales.astype(x.dtype)


def quantize_fp8(t):
    """Per-tensor FP8 (e4m3). Ada (sm_89) supports the dtype natively."""
    amax = jnp.maximum(jnp.max(jnp.abs(t)).astype(jnp.float32), 1e-8)
    scale = amax / 448.0                        # e4m3 max representable
    q = jnp.clip(t.astype(jnp.float32) / scale, -448, 448).astype(jnp.float8_e4m3fn)
    return q, scale


def dequantize_fp8(q, scale, dtype=jnp.bfloat16):
    return (q.astype(jnp.float32) * scale).astype(dtype)


def _is_quantized(x):
    return isinstance(x, dict) and "q" in x and "scales" in x


def quantize_tree(params, skip=("embed", "lm_head", "norm")):
    """int8 every 2-D-or-bigger weight in the pytree. A tree_map, not a walk
    over named_modules -- the model is data here, not a graph of objects.

    Norms are skipped BY NAME, not by rank. jvllm stacks layers, so a norm
    arrives as (num_layers, hidden) -- two-dimensional, and an `ndim >= 2` rule
    quantizes it. They are one vector per layer: nothing to save, and it is
    where the model is most sensitive.
    """
    def visit(path, x):
        name = "/".join(str(getattr(k, "key", getattr(k, "idx", k)))
                        for k in path)
        if any(s in name for s in skip) or jnp.ndim(x) < 2:
            return x
        q, s = quantize_int8_per_channel(x)
        return {"q": q, "scales": s}

    return jax.tree_util.tree_map_with_path(visit, params)


def dequantize_tree(qtree, dtype=jnp.bfloat16):
    return jax.tree.map(
        lambda x: dequantize_int8(x["q"], x["scales"]).astype(dtype)
        if _is_quantized(x) else x,
        qtree, is_leaf=_is_quantized,
    )


def tree_bytes(tree):
    total = 0
    for leaf in jax.tree.leaves(tree, is_leaf=_is_quantized):
        if _is_quantized(leaf):
            total += leaf["q"].size * leaf["q"].dtype.itemsize
            total += leaf["scales"].size * leaf["scales"].dtype.itemsize
        else:
            total += leaf.size * leaf.dtype.itemsize
    return int(total)


def perplexity(model, text, max_len=512):
    ids = model.encode(text)[:max_len]
    logits, _ = model.forward(ids[None], logits_index=None)     # (1, T, V)
    logprobs = jax.nn.log_softmax(logits[0, :-1].astype(jnp.float32), axis=-1)
    targets = ids[1:]
    nll = -jnp.take_along_axis(logprobs, targets[:, None], axis=1).mean()
    return float(jnp.exp(nll))
