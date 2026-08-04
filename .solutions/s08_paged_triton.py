"""Reference solution, stage 08 - paged decode attention in Triton."""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_attn_kernel(
    Out, Q, KC, VC, BT, CTX,
    scale,
    stride_qs, stride_qh,
    stride_kn, stride_kh, stride_kb,
    stride_os, stride_oh,
    stride_bts,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GROUP: tl.constexpr,
):
    s = tl.program_id(0)
    h = tl.program_id(1)
    kvh = h // GROUP

    ctx = tl.load(CTX + s)
    d = tl.arange(0, HEAD_DIM)
    offs = tl.arange(0, BLOCK_SIZE)

    q = tl.load(Q + s * stride_qs + h * stride_qh + d).to(tl.float32) * scale

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    n_blocks = tl.cdiv(ctx, BLOCK_SIZE)
    for b in range(0, n_blocks):
        phys = tl.load(BT + s * stride_bts + b).to(tl.int32)
        pos = b * BLOCK_SIZE + offs
        valid = pos < ctx

        base = phys * stride_kn + kvh * stride_kh + offs[:, None] * stride_kb + d[None, :]
        k = tl.load(KC + base, mask=valid[:, None], other=0.0).to(tl.float32)
        v = tl.load(VC + base, mask=valid[:, None], other=0.0).to(tl.float32)

        scores = tl.sum(q[None, :] * k, axis=1)
        scores = tl.where(valid, scores, float("-inf"))

        # online softmax: rescale the running accumulator, never materialise
        # the full score row
        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

    out = acc / l_i
    tl.store(Out + s * stride_os + h * stride_oh + d, out.to(Out.dtype.element_ty))


def paged_attention_triton(query, key_cache, value_cache, block_tables,
                           context_lens, scale=None):
    S, H, D = query.shape
    _, KVH, BS, _ = key_cache.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    out = torch.empty_like(query)
    ctx = context_lens.to(torch.int32)
    bt = block_tables.to(torch.int32)

    _paged_attn_kernel[(S, H)](
        out, query, key_cache, value_cache, bt, ctx,
        scale,
        query.stride(0), query.stride(1),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        out.stride(0), out.stride(1),
        bt.stride(0),
        HEAD_DIM=D,
        BLOCK_SIZE=BS,
        GROUP=H // KVH,
        num_warps=4,
    )
    return out
