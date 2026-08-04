"""Stage 08 - the same paged attention, fast.

`./vc lore 8` for the insight. `./vc test 8` to check yourself.

Stage 07 was correct and slow: a Python loop over sequences, one gather and one
SDPA call each. This stage collapses it into ONE kernel launch.

The shape of the kernel:

    grid = (num_seqs, num_heads)     one program per (sequence, query head)

    each program:
        load its query vector once           (HEAD_DIM floats, into registers)
        walk its row of the block table
        for each block:
            load a (BLOCK_SIZE, HEAD_DIM) tile of K and of V
            scores = q . K^T
            mask out positions >= context_len
            fold the tile into a running softmax
        write one (HEAD_DIM,) output vector

Online softmax is the trick that lets you do this without ever materialising
the full score row -- the same idea as FlashAttention:

    m_new = max(m_old, max(scores))          running maximum
    alpha = exp(m_old - m_new)               rescale what you already have
    p     = exp(scores - m_new)
    l     = l * alpha + sum(p)               running denominator
    acc   = acc * alpha + sum(p[:, None] * V)
    out   = acc / l

Notes that will save you an afternoon:

  - tl.arange needs a power-of-two length, so HEAD_DIM and BLOCK_SIZE must be
    powers of two (they are: 128 and 16 in practice). Pass them as tl.constexpr
    so Triton specialises the kernel for them.
  - Accumulate in float32 even when the cache is fp16/bf16. Summing hundreds of
    fp16 products loses real precision, and the tolerances will catch it.
  - Mask loads with `pos < context_len` and `other=0.0`. Blocks get reused, so
    slots past the end hold some other request's data.
  - GQA: query head h reads KV head `h // (num_heads // num_kv_heads)`.
  - Pass strides in ELEMENTS (tensor.stride(i)), not bytes.
"""

import math

import torch
import triton
import triton.language as tl


def paged_attention_triton(query, key_cache, value_cache, block_tables,
                           context_lens, scale=None):
    """Same signature and the same numbers as stage 07's paged_attention.

        query        (num_seqs, num_heads, head_dim)
        key_cache    (num_blocks, num_kv_heads, block_size, head_dim)
        value_cache  same
        block_tables (num_seqs, max_blocks_per_seq)
        context_lens (num_seqs,)
        -> (num_seqs, num_heads, head_dim)

    Stage 07 is your oracle. Do not move on until they agree to 1e-3.
    """
    raise NotImplementedError("stage 08: implement paged_attention_triton")
