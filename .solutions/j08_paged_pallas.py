"""Reference solution, stage 08 (jax) - paged decode attention in Pallas."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

from jvllm import compat


def _paged_attn_kernel(q_ref, kc_ref, vc_ref, bt_ref, ctx_ref, o_ref, *,
                       scale, group, block_size, head_dim):
    h = pl.program_id(1)
    kvh = h // group

    # Built INSIDE the kernel. A jnp scalar defined at module level would be
    # captured as a constant, and pallas_call refuses those outright:
    # "captures constants [f32[]]. You should pass them as inputs."
    neg = jnp.float32(-1e30)
    ctx = ctx_ref[0]
    q = q_ref[0, 0].astype(jnp.float32) * scale
    offs = jnp.arange(block_size)

    def body(b, carry):
        m_i, l_i, acc = carry
        phys = bt_ref[0, b]                       # the page-table lookup,
        k = kc_ref[phys, kvh].astype(jnp.float32)  # inside the kernel
        v = vc_ref[phys, kvh].astype(jnp.float32)

        pos = b * block_size + offs
        valid = pos < ctx
        scores = jnp.sum(q[None, :] * k, axis=1)
        scores = jnp.where(valid, scores, neg)

        # online softmax: rescale what you already have, never materialise the
        # full score row
        m_new = jnp.maximum(m_i, jnp.max(scores))
        alpha = jnp.exp(m_i - m_new)
        p = jnp.exp(scores - m_new)
        return (m_new,
                l_i * alpha + jnp.sum(p),
                acc * alpha + jnp.sum(p[:, None] * v, axis=0))

    m_i, l_i, acc = jax.lax.fori_loop(
        0, pl.cdiv(ctx, block_size), body,
        (neg, jnp.float32(0.0), jnp.zeros(head_dim, jnp.float32)),
    )
    o_ref[0, 0] = (acc / l_i).astype(o_ref.dtype)


@functools.partial(jax.jit, static_argnames=("scale",))
def paged_attention_pallas(query, key_cache, value_cache, block_tables,
                           context_lens, scale=None):
    S, H, D = query.shape
    NB, KVH, BS, _ = key_cache.shape
    MAXB = block_tables.shape[1]
    if scale is None:
        scale = float(1.0 / np.sqrt(D))

    kernel = functools.partial(_paged_attn_kernel, scale=scale,
                               group=H // KVH, block_size=BS, head_dim=D)

    # K and V get a BlockSpec covering the WHOLE cache, because which blocks
    # this program needs is not known until it reads the block table. Every
    # other operand is tiled by (seq, head).
    return pl.pallas_call(
        kernel,
        grid=(S, H),
        in_specs=[
            pl.BlockSpec((1, 1, D), lambda s, h: (s, h, 0)),
            pl.BlockSpec((NB, KVH, BS, D), lambda s, h: (0, 0, 0, 0)),
            pl.BlockSpec((NB, KVH, BS, D), lambda s, h: (0, 0, 0, 0)),
            pl.BlockSpec((1, MAXB), lambda s, h: (s, 0)),
            pl.BlockSpec((1,), lambda s, h: (s,)),
        ],
        out_specs=pl.BlockSpec((1, 1, D), lambda s, h: (s, h, 0)),
        out_shape=jax.ShapeDtypeStruct(query.shape, query.dtype),
        compiler_params=compat.compiler_params(num_warps=4),
    )(query, key_cache, value_cache, block_tables,
      context_lens.astype(jnp.int32))
