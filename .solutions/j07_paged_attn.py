"""Reference solution, stage 07 (jax)."""

import jax.numpy as jnp
import numpy as np


def write_kv(key_cache, value_cache, key, value, slot_indices):
    """Functional scatter. Returns NEW caches -- nothing is written in place."""
    block_size = key_cache.shape[2]
    blocks = slot_indices // block_size
    offs = slot_indices % block_size
    # key: (T, kv_heads, head_dim) -> cache: (blocks, kv_heads, block_size, hd)
    return (key_cache.at[blocks, :, offs, :].set(key),
            value_cache.at[blocks, :, offs, :].set(value))


def paged_attention(query, key_cache, value_cache, block_tables,
                    context_lens, scale=None):
    num_seqs, num_heads, head_dim = query.shape
    num_kv_heads, block_size = key_cache.shape[1], key_cache.shape[2]
    group = num_heads // num_kv_heads
    n_blocks = block_tables.shape[1]
    if scale is None:
        scale = 1.0 / np.sqrt(head_dim)

    # One gather for every sequence at once -- no Python loop over sequences.
    # (S, nb, KVH, BS, D) -> (S, KVH, nb*BS, D)
    k = jnp.take(key_cache, block_tables, axis=0)
    v = jnp.take(value_cache, block_tables, axis=0)
    L = n_blocks * block_size
    k = k.transpose(0, 2, 1, 3, 4).reshape(num_seqs, num_kv_heads, L, head_dim)
    v = v.transpose(0, 2, 1, 3, 4).reshape(num_seqs, num_kv_heads, L, head_dim)

    valid = jnp.arange(L)[None, :] < context_lens[:, None]      # (S, L)

    q = query.reshape(num_seqs, num_kv_heads, group, head_dim).astype(jnp.float32)
    scores = jnp.einsum("skgd,skld->skgl", q, k.astype(jnp.float32)) * scale
    scores = jnp.where(valid[:, None, None, :], scores,
                       jnp.finfo(jnp.float32).min)
    probs = jnp.exp(scores - scores.max(axis=-1, keepdims=True))
    probs = probs / probs.sum(axis=-1, keepdims=True)
    out = jnp.einsum("skgl,skld->skgd", probs, v.astype(jnp.float32))
    return out.reshape(num_seqs, num_heads, head_dim).astype(query.dtype)


def reference_attention(query, keys, values, scale=None):
    num_seqs, num_heads, head_dim = query.shape
    num_kv_heads = keys.shape[1]
    group = num_heads // num_kv_heads
    if scale is None:
        scale = 1.0 / np.sqrt(head_dim)

    q = query.reshape(num_seqs, num_kv_heads, group, head_dim).astype(jnp.float32)
    scores = jnp.einsum("skgd,skld->skgl", q, keys.astype(jnp.float32)) * scale
    probs = jnp.exp(scores - scores.max(axis=-1, keepdims=True))
    probs = probs / probs.sum(axis=-1, keepdims=True)
    out = jnp.einsum("skgl,skld->skgd", probs, values.astype(jnp.float32))
    return out.reshape(num_seqs, num_heads, head_dim).astype(query.dtype)
