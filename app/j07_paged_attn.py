"""Stage 07 (JAX) - attention that reads through the page table.

`./vc lore 7 --jax` for the insight. `./vc test 7 --jax` to check yourself.

Stage 06 built the block allocator. The torch track shares it, because it is
pure logic. This stage makes attention read through it. Make it correct here.
Stage 08 makes it fast.

WHAT YOU ARE BUILDING

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
    slot_indices (T,) int32, an ABSOLUTE slot: block_id * block_size + offset

`reference_attention` is the dense oracle: the same numbers computed from
contiguous K and V for each sequence, of shape (seqs, kv_heads, context, D). Both must agree.

WRITE_KV RETURNS THE CACHES

The torch version writes in place and returns nothing. Yours cannot: JAX arrays
are immutable, so a scatter is an expression that produces a new array.

    key_cache.at[block_ids, :, offsets, :].set(key)

Under jit this compiles to an in-place update. Outside jit it really copies.

In torch, a write into the cache looks like an optimization. Here you get it at
no cost while you stay inside jit. You pay a lot for it when you leave jit.
Remember this when you build the engine loop.

NO PYTHON LOOP OVER SEQUENCES

The torch version loops over sequences because indexing a different number of
blocks per sequence is awkward. You do not need to: block_tables is already a
rectangle, so

    keys = jnp.take(key_cache, block_tables, axis=0)
    # (seqs, blocks_per_seq, kv_heads, block_size, D)

gathers every sequence's blocks in one operation. Transpose and reshape it to
(seqs, kv_heads, blocks_per_seq * block_size, D) and you have a dense K per sequence, padded out to
the table width. Then mask positions >= context_len before the softmax.

This is still the SLOW implementation. It makes all of the gathered K and V,
and the whole score row, in memory. But it is slow for a clear reason, and it
is your oracle from now on.

TRAPS

  - Mask with `arange(nb*block_size) < context_len[:, None]`, not with the
    block count. Blocks get REUSED, so the slots past the end of a sequence
    hold some other request's data, not zeros.

  - Softmax in float32 even when the cache is bf16.

  - GQA: query head h reads KV head h // (num_heads // num_kv_heads). A
    reshape of query to (S, num_kv_heads, group, head_dim) gives you exactly
    that, with no repeat_interleave and no copy.

  - The tests SHUFFLE the physical blocks on purpose. An implementation that
    assumes that block b of a sequence lives at physical block b passes no
    check.
"""


def write_kv(key_cache, value_cache, key, value, slot_indices):
    """Scatter the (T, kv_heads, head_dim) K and V into absolute cache
    slots. -> the new (key_cache, value_cache).
    """
    raise NotImplementedError("stage 07 (jax): implement write_kv")


def paged_attention(query, key_cache, value_cache, block_tables,
                    context_lens, scale=None):
    """Decode attention. Gather K and V through the block table."""
    raise NotImplementedError("stage 07 (jax): implement paged_attention")


def reference_attention(query, keys, values, scale=None):
    """The dense oracle: contiguous K and V of shape (seqs, kv_heads,
    context, head_dim)."""
    raise NotImplementedError("stage 07 (jax): implement reference_attention")
