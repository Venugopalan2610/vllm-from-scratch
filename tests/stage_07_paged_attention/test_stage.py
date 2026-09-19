"""Stage 07 - attention that reads through the page table.

The spec is in app/s07_paged_attn.py.

The important contract: paged_attention() must agree with a dense reference
to floating-point tolerance, for every block layout, every context length,
and GQA configurations. If it does not agree, your stage 08 CUDA kernels
have no oracle, and you must debug two things at the same time.
"""

import pytest
import torch

from app.s07_paged_attn import paged_attention, reference_attention, write_kv
from tests.helpers import (
    bench_ms,
    build_paged,
    poison_past_context,
    random_kv,
    random_query,
)

TOLERANCE = dict(rtol=2e-3, atol=2e-3)


def _assert_matches_reference(num_seqs, num_heads, num_kv_heads, head_dim,
                              context_len, block_size, device):
    keys, values = random_kv(num_seqs, num_kv_heads, context_len, head_dim,
                             device)
    query = random_query(num_seqs, num_heads, head_dim, device)
    expected = reference_attention(query, keys, values)
    output = paged_attention(query, *build_paged(keys, values, block_size))
    assert output.shape == expected.shape
    torch.testing.assert_close(output, expected, **TOLERANCE)


# ---- write_kv -------------------------------------------------------

def test_write_kv_scatters_to_the_right_slots(device):
    block_size, head_dim, num_kv_heads = 4, 8, 2
    key_cache = torch.zeros(6, num_kv_heads, block_size, head_dim, device=device)
    value_cache = torch.zeros_like(key_cache)
    key = torch.randn(3, num_kv_heads, head_dim, device=device)
    value = torch.randn(3, num_kv_heads, head_dim, device=device)
    # slots 5, 0, 21 -> (block 1, offset 1), (block 0, offset 0), (block 5, offset 1)
    write_kv(key_cache, value_cache, key, value,
             torch.tensor([5, 0, 21], device=device))

    assert torch.allclose(key_cache[1, :, 1], key[0])
    assert torch.allclose(key_cache[0, :, 0], key[1])
    assert torch.allclose(key_cache[5, :, 1], key[2])
    assert torch.allclose(value_cache[5, :, 1], value[2])
    assert key_cache[2].abs().sum() == 0, "wrote to a block that was not a target"


# ---- correctness ----------------------------------------------------

@pytest.mark.parametrize("block_size", [1, 4, 16])
@pytest.mark.parametrize("context_len", [1, 7, 16, 33])
def test_matches_dense_reference(device, block_size, context_len):
    """Every block size, also a context that is not a multiple of it."""
    _assert_matches_reference(3, 4, 4, 16, context_len, block_size, device)


def test_gqa(device):
    """num_heads != num_kv_heads. Query head h reads KV head h // group.

    If the ratio is the wrong way, the output still looks like attention
    output. It is only wrong. That is the most frequent bug of this stage.
    """
    _assert_matches_reference(2, 8, 2, 16, 20, 8, device)     # group = 4


def test_ragged_context_lengths(device):
    """The sequences of one batch have different lengths. That is the usual
    case."""
    num_seqs, num_heads, num_kv_heads, head_dim, longest = 4, 4, 2, 16, 40
    keys, values = random_kv(num_seqs, num_kv_heads, longest, head_dim, device)
    key_cache, value_cache, block_tables, _ = build_paged(keys, values, 8)
    context_lens = torch.tensor([40, 1, 17, 32], dtype=torch.int32,
                                device=device)
    query = random_query(num_seqs, num_heads, head_dim, device)

    output = paged_attention(query, key_cache, value_cache, block_tables,
                             context_lens)
    for seq, context_len in enumerate(context_lens.tolist()):
        expected = reference_attention(query[seq:seq + 1],
                                       keys[seq:seq + 1, :, :context_len],
                                       values[seq:seq + 1, :, :context_len])
        torch.testing.assert_close(output[seq:seq + 1], expected, **TOLERANCE)


def test_ignores_junk_beyond_context_len(device):
    """The allocator gives a freed block to another sequence, and that
    sequence writes into it.

    If you attend past context_lens, you mix in the data of a different
    request with no error. That is a correctness bug AND a data leak between
    users.
    """
    num_seqs, num_heads, num_kv_heads, head_dim, context_len = 2, 4, 2, 16, 12
    block_size = 8
    keys, values = random_kv(num_seqs, num_kv_heads, context_len, head_dim,
                             device)
    key_cache, value_cache, block_tables, _ = build_paged(keys, values,
                                                          block_size)
    context_lens = torch.tensor([context_len] * num_seqs, dtype=torch.int32,
                                device=device)
    query = random_query(num_seqs, num_heads, head_dim, device)

    def attend():
        return paged_attention(query, key_cache, value_cache, block_tables,
                               context_lens)

    before = attend()
    poison_past_context(key_cache, value_cache, block_tables, context_len,
                        block_size)
    torch.testing.assert_close(before, attend(), **TOLERANCE)


def test_physical_block_order_is_irrelevant(device):
    """The same logical content in a different physical layout -> the same
    output."""
    keys, values = random_kv(2, 4, 24, 16, device)
    query = random_query(2, 4, 16, device)
    first = paged_attention(query, *build_paged(keys, values, 8, seed=1))
    second = paged_attention(query, *build_paged(keys, values, 8, seed=2))
    torch.testing.assert_close(first, second, **TOLERANCE)


def test_softmax_rows_sum_to_one(device):
    """A cheap invariant that finds masking bugs that the reference can
    share."""
    keys, values = random_kv(2, 4, 16, 8, device)
    key_cache, value_cache, block_tables, context_lens = build_paged(
        keys, values, 4)
    # With V = all ones, the attention output must be exactly 1 everywhere.
    output = paged_attention(random_query(2, 4, 8, device), key_cache,
                             torch.ones_like(value_cache), block_tables,
                             context_lens)
    torch.testing.assert_close(output, torch.ones_like(output),
                               rtol=1e-3, atol=1e-3)


def test_report_the_slowdown(device):
    """Not a pass or a fail. Paging costs you time. Find out how much."""
    num_seqs, num_heads, num_kv_heads, head_dim, context_len = 64, 16, 8, 128, 512
    keys, values = random_kv(num_seqs, num_kv_heads, context_len, head_dim,
                             device, dtype=torch.float16)
    query = random_query(num_seqs, num_heads, head_dim, device, torch.float16)
    paged_cache = build_paged(keys, values, 16)

    dense_ms = bench_ms(lambda: reference_attention(query, keys, values),
                        iters=10, warmup=3)
    paged_ms = bench_ms(lambda: paged_attention(query, *paged_cache),
                        iters=10, warmup=3)
    print(f"\n  dense reference: {dense_ms:7.2f} ms")
    print(f"  paged (PyTorch): {paged_ms:7.2f} ms   "
          f"({paged_ms / dense_ms:.1f}x slower)")
    print("\n  \033[2mThe Python loop over the sequences is the problem.")
    print("  Stage 08 replaces it with one CUDA kernel and gets this")
    print("  back.\033[0m")
