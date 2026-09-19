"""Stage 07 (JAX) - paged attention, the correct and slow reference.

The spec is in app/j07_paged_attn.py.

Every fixture here shuffles the physical blocks. An implementation that
thinks that sequence block b is at physical block b passes no check.
"""

import jax.numpy as jnp
import pytest

from app.j07_paged_attn import paged_attention, reference_attention, write_kv
from tests.jhelpers import (
    assert_close,
    build_paged,
    jbench_ms,
    poisoned_past_context,
    random_kv,
    random_query,
)


def test_write_kv_scatters_to_the_right_slots(jax_device):
    """A slot is block_id * block_size + offset, and the blocks are not in
    order."""
    num_blocks, num_kv_heads, block_size, head_dim = 6, 2, 4, 8
    key_cache = jnp.zeros((num_blocks, num_kv_heads, block_size, head_dim))
    value_cache = jnp.zeros_like(key_cache)
    slots = [0, 5, 9, 23]                          # blocks 0, 1, 2, 5
    key = jnp.arange(4 * num_kv_heads * head_dim,
                     dtype=jnp.float32).reshape(4, num_kv_heads, head_dim)
    value = -key

    key_cache, value_cache = write_kv(key_cache, value_cache, key, value,
                                      jnp.asarray(slots, jnp.int32))
    for token, slot in enumerate(slots):
        block_id, offset = slot // block_size, slot % block_size
        assert_close(key_cache[block_id, :, offset, :], key[token])
        assert_close(value_cache[block_id, :, offset, :], value[token])
    assert float(jnp.abs(key_cache).sum()) > 0, "wrote nothing"
    assert float(jnp.abs(key_cache[3]).sum()) == 0.0, (
        "wrote into a block that was not a target")


@pytest.mark.parametrize("block_size", [1, 4, 16])
@pytest.mark.parametrize("context_len", [1, 7, 16, 33, 129])
def test_matches_the_dense_reference(jax_device, block_size, context_len):
    """Every layout, every length, against contiguous K and V."""
    keys, values = random_kv(3, 4, context_len, 16)
    query = random_query(3, 4, 16, seed=0)
    assert_close(paged_attention(query, *build_paged(keys, values, block_size)),
                 reference_attention(query, keys, values))


@pytest.mark.parametrize("num_heads,num_kv_heads", [(8, 2), (16, 8), (4, 4)])
def test_gqa_ratios(jax_device, num_heads, num_kv_heads):
    """Query head h must read KV head h // (num_heads // num_kv_heads)."""
    keys, values = random_kv(2, num_kv_heads, 40, 64)
    query = random_query(2, num_heads, 64, seed=1)
    assert_close(paged_attention(query, *build_paged(keys, values, 16)),
                 reference_attention(query, keys, values))


def test_ragged_context_lengths(jax_device):
    """Different lengths in one call, all through the same block table."""
    num_seqs = 8
    keys, values = random_kv(num_seqs, 4, 200, 64)
    key_cache, value_cache, block_tables, _ = build_paged(keys, values, 16)
    context_lens = jnp.asarray([200, 1, 17, 16, 199, 33, 64, 128], jnp.int32)
    query = random_query(num_seqs, 8, 64, seed=2)

    output = paged_attention(query, key_cache, value_cache, block_tables,
                             context_lens)
    for seq in range(num_seqs):
        context_len = int(context_lens[seq])
        expected = reference_attention(query[seq:seq + 1],
                                       keys[seq:seq + 1, :, :context_len],
                                       values[seq:seq + 1, :, :context_len])
        assert_close(output[seq:seq + 1], expected)


def test_ignores_junk_beyond_context_len(jax_device):
    """The allocator reuses blocks, so the end of the last block holds the
    data of another sequence.

    Mask on `position < context_len`, not on the number of blocks.
    """
    context_len, block_size = 12, 8
    keys, values = random_kv(2, 2, context_len, 32)
    key_cache, value_cache, block_tables, _ = build_paged(keys, values,
                                                          block_size)
    context_lens = jnp.asarray([context_len, context_len], jnp.int32)
    query = random_query(2, 4, 32, seed=3)

    before = paged_attention(query, key_cache, value_cache, block_tables,
                             context_lens)
    after = paged_attention(query, *poisoned_past_context(
        key_cache, value_cache, block_tables, context_len, block_size),
        block_tables, context_lens)
    assert_close(after, before)


def test_bf16_cache_accumulates_in_fp32(jax_device):
    """A long bf16 context shows careless accumulation."""
    keys, values = random_kv(2, 8, 1024, 128, dtype=jnp.bfloat16)
    query = random_query(2, 8, 128, seed=4, dtype=jnp.bfloat16)
    expected = reference_attention(query.astype(jnp.float32),
                                   keys.astype(jnp.float32),
                                   values.astype(jnp.float32))
    assert_close(paged_attention(query, *build_paged(keys, values, 16)),
                 expected, tolerance=6e-2)


def test_the_slowdown_you_just_ate(jax_device):
    """Not a pass or a fail. Paging costs you time. See how much."""
    keys, values = random_kv(32, 8, 512, 128, dtype=jnp.bfloat16)
    query = random_query(32, 16, 128, seed=5, dtype=jnp.bfloat16)
    paged_cache = build_paged(keys, values, 16)

    dense_ms = jbench_ms(lambda: reference_attention(query, keys, values))
    paged_ms = jbench_ms(lambda: paged_attention(query, *paged_cache))
    print(f"\n  dense (contiguous K/V): {dense_ms:7.3f} ms")
    print(f"  paged (gather first):  {paged_ms:7.3f} ms   "
          f"{paged_ms / dense_ms:.2f}x")
    print("\n  \033[2mThe gather is a full copy of the K and V of every")
    print("  sequence, written to HBM and read back. Stage 08 removes it: it")
    print("  never writes the gathered cache to memory.\033[0m")
