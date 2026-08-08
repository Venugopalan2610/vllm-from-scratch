"""Stage 07 (JAX) - attention that reads through the page table.

`./vc lore 7 --jax` for the insight. `./vc test 7 --jax` to check yourself.

Stage 06 built the block allocator (shared with the torch track -- it is pure
logic). This stage makes attention read through it. Correct first, fast in
stage 08.

WHAT YOU'RE BUILDING

    write_kv(key_cache, value_cache, key, value, slot_indices)
        -> (key_cache, value_cache)
    paged_attention(query, key_cache, value_cache, block_tables,
                    context_lens, scale=None) -> (S, H, D)
    reference_attention(query, keys, values, scale=None) -> (S, H, D)

    query        (num_seqs, num_heads, head_dim)
    key_cache    (num_blocks, num_kv_heads, block_size, head_dim)
    value_cache  same
    block_tables (num_seqs, max_blocks_per_seq) int32
    context_lens (num_seqs,) int32
    slot_indices (T,) int32 -- an ABSOLUTE slot, block*block_size + offset

`reference_attention` is the dense oracle: the same numbers computed from
contiguous per-sequence K/V of shape (S, KVH, L, D). Both must agree.

WRITE_KV RETURNS THE CACHES

The torch version writes in place and returns nothing. Yours cannot: JAX arrays
are immutable, so a scatter is an expression that produces a new array.

    key_cache.at[blocks, :, offs, :].set(key)

Under jit this compiles to an in-place update; outside jit it really does copy.
Which means the thing that looks like an optimization in torch -- mutating the
cache -- is a thing you get for free here as long as you stay inside jit, and
pay for dearly if you drop out of it. Worth remembering when you build the
engine loop.

NO PYTHON LOOP OVER SEQUENCES

The torch version loops over sequences because indexing a different number of
blocks per sequence is awkward. You do not need to: block_tables is already a
rectangle, so

    k = jnp.take(key_cache, block_tables, axis=0)   # (S, nb, KVH, BS, D)

gathers every sequence's blocks in one operation. Transpose and reshape it to
(S, KVH, nb*block_size, D) and you have a dense K per sequence, padded out to
the table width. Then mask positions >= context_len before the softmax.

This is still the SLOW implementation -- it materialises the entire gathered
K/V and the entire score row -- but it is slow for an honest reason, and it is
your oracle forever after.

TRAPS

  - Mask with `arange(nb*block_size) < context_len[:, None]`, not with the
    block count. Blocks get REUSED, so the slots past the end of a sequence
    hold some other request's data, not zeros.

  - Softmax in float32 even when the cache is bf16.

  - GQA: query head h reads KV head h // (num_heads // num_kv_heads). Reshaping
    query to (S, num_kv_heads, group, head_dim) gives you exactly that, with no
    repeat_interleave and no copy.

  - The physical blocks are SHUFFLED on purpose in the tests. If your
    implementation assumes block b of a sequence lives at physical block b, it
    will pass nothing.
"""


def write_kv(key_cache, value_cache, key, value, slot_indices):
    """Scatter (T, kv_heads, head_dim) K/V into absolute cache slots.

    Returns the updated (key_cache, value_cache).
    """
    raise NotImplementedError("stage 07 (jax): implement write_kv")


def paged_attention(query, key_cache, value_cache, block_tables,
                    context_lens, scale=None):
    """Decode attention, gathering K/V through the block table."""
    raise NotImplementedError("stage 07 (jax): implement paged_attention")


def reference_attention(query, keys, values, scale=None):
    """The dense oracle: contiguous K/V of shape (S, kv_heads, L, head_dim)."""
    raise NotImplementedError("stage 07 (jax): implement reference_attention")
