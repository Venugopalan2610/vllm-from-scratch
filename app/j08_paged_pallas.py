"""Stage 08 (JAX) - the same paged attention, fast, in Pallas.

`./vc lore 8 --jax` for the insight. `./vc test 8 --jax` to check yourself.

Stage 07 was correct and slow. It gathered the blocks of every sequence into
a dense K and V, wrote them to HBM, and read them back. This stage puts all
of that into ONE kernel launch that never writes the gather to memory.

    paged_attention_pallas(query, key_cache, value_cache, block_tables,
                           context_lens, scale=None) -> (S, H, D)

The same signature and the same numbers as stage 07. Stage 07 is your
oracle. Do not continue until they agree to 1e-3.

THE SHAPE OF THE KERNEL

    grid = (num_seqs, num_heads)     one program for each (sequence, query head)

    each program:
        load its query vector one time       (head_dim floats, into registers)
        walk its row of the block table
        for each block:
            load a (block_size, head_dim) tile of K and of V
            scores = q . K^T
            mask the positions >= context_len
            add the tile to a running softmax
        write one (head_dim,) output vector

The online softmax lets you do that, and you never hold the full score row.
It is the idea of FlashAttention:

    new_max     = max(running_max, max(scores))
    rescale     = exp(running_max - new_max)      rescale what you have
    p           = exp(scores - new_max)
    running_sum = running_sum * rescale + sum(p)  the denominator
    accumulator = accumulator * rescale + sum(p[:, None] * V)
    output      = accumulator / running_sum

PALLAS IS NOT TRITON

Triton gives you raw pointers, and you do the address arithmetic yourself:
`key_cache + block_id * stride_n + kv_head * stride_h + ...`. Pallas gives
you Refs, and a BlockSpec selects the tile of the array that the Ref of each
program points at:

    pl.BlockSpec(block_shape, index_map)

    index_map gets the grid indices and returns the BLOCK of the array that
    this program sees. So the query gets `pl.BlockSpec((1, 1, D),
    lambda seq, head: (seq, head, 0))`. Program (seq, head) sees only its
    own query vector, and in the kernel that is `query_ref[0, 0]`.

The KV cache does NOT fit that model, and that is the point of the stage.
The program does not know WHICH blocks it needs until it reads the block
table, inside the kernel. So K and V get a BlockSpec that covers the whole
cache, and you index it at run time:

    block_id = block_table_ref[0, block_index]
    keys = key_cache_ref[block_id, kv_head]      # (block_size, head_dim)

That is the paged indirection, and no BlockSpec can express it.

TRAPS

  - `jnp.arange(block_size)` inside a kernel needs a length that is a power
    of two, and head_dim does too. Both are (16, 4, 1 and 128, 64, 32, 16 in
    the checks). Give them as Python ints that the kernel closes over, not
    as traced values. They must be constants at compile time.

  - The number of blocks is DYNAMIC: it depends on context_len, which is a
    device value. Use `jax.lax.fori_loop(0, pl.cdiv(context_len,
    block_size), ...)`. A Python `for` needs the trip count at trace time,
    and you do not have it.

  - Make your -inf constant INSIDE the kernel. pallas_call captures a `jnp`
    scalar from module level as a constant, and it refuses it with
    "captures constants [f32[]]. You should pass them as inputs."

  - Accumulate in float32, also when the cache is bf16. A sum of hundreds
    of bf16 products loses real precision, and the tolerances find it.

  - Mask on `position < context_len`. The allocator reuses blocks, so the
    slots after the end hold the data of another request, not zeros.

  - GQA: query head h reads KV head `h // (num_heads // num_kv_heads)`.

  - On a consumer GPU, the default Mosaic backend of Pallas cannot compile
    this. `jvllm.compat.compiler_params()` returns the Triton-backend params
    that can. Give it as `compiler_params=`. jvllm/compat.py tells why.
"""


def paged_attention_pallas(query, key_cache, value_cache, block_tables,
                           context_lens, scale=None):
    """The same signature and the same numbers as the paged_attention of
    stage 07.

        query        (num_seqs, num_heads, head_dim)
        key_cache    (num_blocks, num_kv_heads, block_size, head_dim)
        value_cache  same
        block_tables (num_seqs, max_blocks_per_seq)
        context_lens (num_seqs,)
        -> (num_seqs, num_heads, head_dim)
    """
    raise NotImplementedError("stage 08 (jax): implement paged_attention_pallas")
