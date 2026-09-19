"""Reference solution, stage 07 (jax)."""

import jax.numpy as jnp
import numpy as np


def write_kv(key_cache, value_cache, key, value, slot_indices):
    """A functional scatter. It returns NEW caches, and it writes nothing in
    place. key: (tokens, kv_heads, D). A cache: (blocks, kv_heads,
    block_size, D)."""
    block_size = key_cache.shape[2]
    block_ids = slot_indices // block_size
    offsets = slot_indices % block_size
    return (key_cache.at[block_ids, :, offsets, :].set(key),
            value_cache.at[block_ids, :, offsets, :].set(value))


def gather_all_contexts(cache, block_tables):
    """One gather for all sequences, with no Python loop.
    (seqs, blocks, kv_heads, block_size, D) -> (seqs, kv_heads, slots, D)."""
    num_seqs, blocks_per_seq = block_tables.shape
    _, num_kv_heads, block_size, head_dim = cache.shape
    rows = jnp.take(cache, block_tables, axis=0).transpose(0, 2, 1, 3, 4)
    return rows.reshape(num_seqs, num_kv_heads, blocks_per_seq * block_size,
                        head_dim)


def grouped_attention(query, keys, values, scale, valid=None):
    """query (seqs, heads, D). keys and values (seqs, kv_heads, context, D).
    valid (seqs, context) hides the slots past each context."""
    num_seqs, num_heads, head_dim = query.shape
    num_kv_heads = keys.shape[1]
    grouped = query.reshape(num_seqs, num_kv_heads, num_heads // num_kv_heads,
                            head_dim).astype(jnp.float32)
    scores = jnp.einsum("skgd,skld->skgl", grouped,
                        keys.astype(jnp.float32)) * scale
    if valid is not None:
        scores = jnp.where(valid[:, None, None, :], scores,
                           jnp.finfo(jnp.float32).min)
    probs = jnp.exp(scores - scores.max(axis=-1, keepdims=True))
    probs = probs / probs.sum(axis=-1, keepdims=True)
    output = jnp.einsum("skgl,skld->skgd", probs, values.astype(jnp.float32))
    return output.reshape(num_seqs, num_heads, head_dim).astype(query.dtype)


def paged_attention(query, key_cache, value_cache, block_tables,
                    context_lens, scale=None):
    scale = scale or 1.0 / np.sqrt(query.shape[2])
    keys = gather_all_contexts(key_cache, block_tables)
    values = gather_all_contexts(value_cache, block_tables)
    valid = jnp.arange(keys.shape[2])[None, :] < context_lens[:, None]
    return grouped_attention(query, keys, values, scale, valid)


def reference_attention(query, keys, values, scale=None):
    scale = scale or 1.0 / np.sqrt(query.shape[2])
    return grouped_attention(query, keys, values, scale)
