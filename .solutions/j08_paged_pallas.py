"""Reference solution, stage 08 (jax) - paged decode attention in Pallas."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

from jvllm import compat


def _paged_attention_kernel(query_ref, key_cache_ref, value_cache_ref,
                            block_table_ref, context_len_ref, output_ref, *,
                            scale, heads_per_kv_head, block_size, head_dim):
    kv_head = pl.program_id(1) // heads_per_kv_head
    # Make the constant INSIDE the kernel. pallas_call refuses a jnp scalar
    # that it captures from module level: "captures constants [f32[]]. You
    # should pass them as inputs."
    masked_score = jnp.float32(-1e30)
    context_len = context_len_ref[0]
    query = query_ref[0, 0].astype(jnp.float32) * scale
    offsets = jnp.arange(block_size)

    def attend_block(block_index, carry):
        running_max, running_sum, accumulator = carry
        block_id = block_table_ref[0, block_index]      # the page table read,
        keys = key_cache_ref[block_id, kv_head].astype(jnp.float32)  # inside
        values = value_cache_ref[block_id, kv_head].astype(jnp.float32)

        valid = block_index * block_size + offsets < context_len
        scores = jnp.where(valid, jnp.sum(query[None, :] * keys, axis=1),
                           masked_score)
        # Online softmax: rescale what you have. Never make the full score row.
        new_max = jnp.maximum(running_max, jnp.max(scores))
        rescale = jnp.exp(running_max - new_max)
        probs = jnp.exp(scores - new_max)
        return (new_max,
                running_sum * rescale + jnp.sum(probs),
                accumulator * rescale + jnp.sum(probs[:, None] * values,
                                                axis=0))

    _, total, accumulator = jax.lax.fori_loop(
        0, pl.cdiv(context_len, block_size), attend_block,
        (masked_score, jnp.float32(0.0), jnp.zeros(head_dim, jnp.float32)))
    output_ref[0, 0] = (accumulator / total).astype(output_ref.dtype)


@functools.partial(jax.jit, static_argnames=("scale",))
def paged_attention_pallas(query, key_cache, value_cache, block_tables,
                           context_lens, scale=None):
    num_seqs, num_heads, head_dim = query.shape
    num_blocks, num_kv_heads, block_size, _ = key_cache.shape
    blocks_per_seq = block_tables.shape[1]
    scale = scale or float(1.0 / np.sqrt(head_dim))

    kernel = functools.partial(_paged_attention_kernel, scale=scale,
                               heads_per_kv_head=num_heads // num_kv_heads,
                               block_size=block_size, head_dim=head_dim)
    one_head = pl.BlockSpec((1, 1, head_dim), lambda seq, head: (seq, head, 0))
    # K and V get a BlockSpec over the WHOLE cache. A program knows its
    # blocks only after it reads the block table. The other operands get a
    # (seq, head) tile.
    whole_cache = pl.BlockSpec((num_blocks, num_kv_heads, block_size,
                                head_dim), lambda seq, head: (0, 0, 0, 0))
    return pl.pallas_call(
        kernel,
        grid=(num_seqs, num_heads),
        in_specs=[one_head, whole_cache, whole_cache,
                  pl.BlockSpec((1, blocks_per_seq),
                               lambda seq, head: (seq, 0)),
                  pl.BlockSpec((1,), lambda seq, head: (seq,))],
        out_specs=one_head,
        out_shape=jax.ShapeDtypeStruct(query.shape, query.dtype),
        compiler_params=compat.compiler_params(num_warps=4),
    )(query, key_cache, value_cache, block_tables,
      context_lens.astype(jnp.int32))
