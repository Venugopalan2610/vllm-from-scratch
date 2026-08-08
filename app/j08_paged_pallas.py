"""Stage 08 (JAX) - the same paged attention, fast, in Pallas.

`./vc lore 8 --jax` for the insight. `./vc test 8 --jax` to check yourself.

Stage 07 was correct and slow: it gathered every sequence's blocks into a dense
K and V, wrote them to HBM, and read them straight back. This stage collapses
the whole thing into ONE kernel launch that never materialises the gather.

    paged_attention_pallas(query, key_cache, value_cache, block_tables,
                           context_lens, scale=None) -> (S, H, D)

Same signature and the same numbers as stage 07. Stage 07 is your oracle. Do
not move on until they agree to 1e-3.

THE SHAPE OF THE KERNEL

    grid = (num_seqs, num_heads)     one program per (sequence, query head)

    each program:
        load its query vector once           (head_dim floats, into registers)
        walk its row of the block table
        for each block:
            load a (block_size, head_dim) tile of K and of V
            scores = q . K^T
            mask out positions >= context_len
            fold the tile into a running softmax
        write one (head_dim,) output vector

Online softmax is what lets you do that without ever holding the full score
row -- the same idea as FlashAttention:

    m_new = max(m_old, max(scores))          running maximum
    alpha = exp(m_old - m_new)               rescale what you already have
    p     = exp(scores - m_new)
    l     = l * alpha + sum(p)               running denominator
    acc   = acc * alpha + sum(p[:, None] * V)
    out   = acc / l

PALLAS IS NOT TRITON, QUITE

Triton hands you raw pointers and you do the address arithmetic yourself:
`KC + phys*stride_n + kvh*stride_h + ...`. Pallas hands you Refs, and a
BlockSpec decides which tile of the array each program's Ref points at:

    pl.BlockSpec(block_shape, index_map)

    index_map is called with the grid indices and returns which BLOCK of the
    array this program sees. So q gets `pl.BlockSpec((1, 1, D), lambda s, h:
    (s, h, 0))` -- program (s, h) sees exactly its own query vector, and inside
    the kernel that is just `q_ref[0, 0]`.

The one thing that does NOT fit that model is the KV cache, and it is the whole
point of the stage: WHICH blocks this program needs is not known until it reads
the block table, which happens inside the kernel. So K and V get a BlockSpec
covering the entire cache, and you index it dynamically:

    phys = bt_ref[0, b]
    k = kc_ref[phys, kvh]        # (block_size, head_dim)

That is the paged indirection, and there is no BlockSpec that expresses it.

NOTES THAT WILL SAVE YOU AN AFTERNOON

  - `jnp.arange(block_size)` inside a kernel needs a power-of-two length, and
    head_dim does too. Both are (16, 4, 1 and 128, 64, 32, 16 in the tests).
    Pass them as Python ints closed over by the kernel, not as traced values --
    they have to be compile-time constants.

  - The block count is DYNAMIC: it depends on context_len, which is a device
    value. Use `jax.lax.fori_loop(0, pl.cdiv(ctx, block_size), ...)`. A Python
    `for` would need the trip count at trace time and you do not have it.

  - Build your -inf sentinel INSIDE the kernel. A `jnp` scalar defined at
    module level gets captured as a constant, and pallas_call rejects that
    with "captures constants [f32[]]. You should pass them as inputs."

  - Accumulate in float32 even when the cache is bf16. Summing hundreds of
    bf16 products loses real precision and the tolerances will catch it.

  - Mask on `pos < context_len`. Blocks get recycled, so slots past the end
    hold another request's data -- not zeros.

  - GQA: query head h reads KV head `h // (num_heads // num_kv_heads)`.

  - On a consumer GPU, Pallas's default Mosaic backend cannot compile this;
    `jvllm.compat.compiler_params()` returns the Triton-backend params that
    can. Pass it through as `compiler_params=`. See jvllm/compat.py for why.
"""


def paged_attention_pallas(query, key_cache, value_cache, block_tables,
                           context_lens, scale=None):
    """Same signature and the same numbers as stage 07's paged_attention.

        query        (num_seqs, num_heads, head_dim)
        key_cache    (num_blocks, num_kv_heads, block_size, head_dim)
        value_cache  same
        block_tables (num_seqs, max_blocks_per_seq)
        context_lens (num_seqs,)
        -> (num_seqs, num_heads, head_dim)
    """
    raise NotImplementedError("stage 08 (jax): implement paged_attention_pallas")
