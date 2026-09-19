"""Stage 08 (JAX) - paged attention in Pallas.

The spec is in app/j08_paged_pallas.py.

Every correctness check compares with stage 07, which is your oracle. The
last check is the important one: it must really be faster.

These checks need a Pallas GPU backend that works. On a card older than
Hopper, that is the Triton backend, and `jvllm.compat` registers it for you.
If no backend can compile here, the whole file skips. It does not fail.
"""

import jax.numpy as jnp
import pytest

from app.j07_paged_attn import paged_attention, reference_attention
from app.j08_paged_pallas import paged_attention_pallas
from tests.jhelpers import (
    assert_close,
    build_paged,
    jbench_ms,
    poisoned_past_context,
    random_kv,
    random_query,
)


@pytest.mark.parametrize("block_size", [1, 4, 16])
@pytest.mark.parametrize("context_len", [1, 7, 16, 33, 129])
def test_agrees_with_stage_07(jpallas, block_size, context_len):
    """The same numbers as the jnp version, every layout, every length."""
    keys, values = random_kv(3, 4, context_len, 16)
    query = random_query(3, 4, 16, seed=0)
    paged_cache = build_paged(keys, values, block_size)
    assert_close(paged_attention_pallas(query, *paged_cache),
                 paged_attention(query, *paged_cache))


def test_agrees_with_the_dense_reference_too(jpallas):
    """A second check: not only the same as stage 07, but correct."""
    keys, values = random_kv(4, 8, 100, 64)
    query = random_query(4, 8, 64, seed=1)
    assert_close(paged_attention_pallas(query, *build_paged(keys, values, 16)),
                 reference_attention(query, keys, values))


@pytest.mark.parametrize("num_heads,num_kv_heads", [(8, 2), (16, 8), (4, 4)])
def test_gqa_ratios(jpallas, num_heads, num_kv_heads):
    """Query head h must read KV head h // (num_heads // num_kv_heads)."""
    keys, values = random_kv(2, num_kv_heads, 40, 64)
    query = random_query(2, num_heads, 64, seed=2)
    paged_cache = build_paged(keys, values, 16)
    assert_close(paged_attention_pallas(query, *paged_cache),
                 paged_attention(query, *paged_cache))


def test_ragged_context_lengths(jpallas):
    """The trip count of the fori_loop is different for each program, and
    it is a device value."""
    keys, values = random_kv(8, 4, 200, 64)
    key_cache, value_cache, block_tables, _ = build_paged(keys, values, 16)
    context_lens = jnp.asarray([200, 1, 17, 16, 199, 33, 64, 128], jnp.int32)
    query = random_query(8, 8, 64, seed=3)
    arguments = (query, key_cache, value_cache, block_tables, context_lens)
    assert_close(paged_attention_pallas(*arguments),
                 paged_attention(*arguments))


def test_ignores_junk_beyond_context_len(jpallas):
    """The masking check again, because a kernel is much easier to get
    wrong."""
    context_len, block_size = 12, 8
    keys, values = random_kv(2, 2, context_len, 32)
    key_cache, value_cache, block_tables, _ = build_paged(keys, values,
                                                          block_size)
    context_lens = jnp.asarray([context_len, context_len], jnp.int32)
    query = random_query(2, 4, 32, seed=4)

    before = paged_attention_pallas(query, key_cache, value_cache,
                                    block_tables, context_lens)
    after = paged_attention_pallas(query, *poisoned_past_context(
        key_cache, value_cache, block_tables, context_len, block_size),
        block_tables, context_lens)
    assert_close(after, before)


def test_bf16_accumulates_in_fp32(jpallas):
    """A long context in bf16 shows careless accumulation.

    A bf16 sum of 2048 bf16 products loses several digits. Accumulate in
    float32 inside the kernel, and this passes easily.
    """
    keys, values = random_kv(2, 8, 2048, 128, dtype=jnp.bfloat16)
    query = random_query(2, 8, 128, seed=5, dtype=jnp.bfloat16)
    expected = reference_attention(query.astype(jnp.float32),
                                   keys.astype(jnp.float32),
                                   values.astype(jnp.float32))
    assert_close(paged_attention_pallas(query, *build_paged(keys, values, 16)),
                 expected, tolerance=6e-2)


def test_it_is_actually_faster(jpallas):
    """The point of the stage."""
    rows = []
    for num_seqs, context_len in ((8, 256), (32, 512), (64, 1024)):
        keys, values = random_kv(num_seqs, 8, context_len, 128,
                                 dtype=jnp.bfloat16)
        query = random_query(num_seqs, 16, 128, seed=6, dtype=jnp.bfloat16)
        paged_cache = build_paged(keys, values, 16)
        jnp_ms = jbench_ms(lambda: paged_attention(query, *paged_cache),
                           iters=10)
        pallas_ms = jbench_ms(lambda: paged_attention_pallas(query,
                                                             *paged_cache))
        rows.append((num_seqs, context_len, jnp_ms, pallas_ms))

    print(f"\n  {'seqs':>5} {'ctx':>6} {'stage 07':>11} {'stage 08':>11} "
          f"{'speedup':>9}")
    for num_seqs, context_len, jnp_ms, pallas_ms in rows:
        print(f"  {num_seqs:>5} {context_len:>6} {jnp_ms:>9.2f}ms "
              f"{pallas_ms:>9.3f}ms {jnp_ms / pallas_ms:>8.1f}x")

    worst = min(jnp_ms / pallas_ms for _, _, jnp_ms, pallas_ms in rows)
    assert worst > 3.0, (
        f"only {worst:.1f}x faster in the worst case. Stage 07 copies the "
        "whole K and V of every sequence to HBM before it computes anything. "
        "One kernel that streams tiles must be much faster.")
    print(f"\n  \033[1mWorst case: {worst:.0f}x faster than the jnp "
          "version.\033[0m")
    print("  \033[2mYou removed the gather. The K and V tiles now go through")
    print("  SRAM, not through a copy in HBM, and the score row never")
    print("  exists.\033[0m")
