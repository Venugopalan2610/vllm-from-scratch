"""Stage 08 - paged attention in CUDA.

The spec is in app/cuda/s08_paged_attn.cu and app/s08_paged_cuda.py.

Every correctness check here compares with stage 07, which is your oracle.
The last two checks are the most important. The kernel must be faster than
the PyTorch version. And the bandwidth that it reports is the number that
stage 08b changes.
"""

import pytest
import torch

from app.s07_paged_attn import paged_attention, reference_attention, write_kv
from app.s08_paged_cuda import paged_attention_cuda, write_kv_cuda
from tests.helpers import (
    bench_ms,
    kv_bytes,
    paged_problem,
    poison_past_context,
    random_query,
)

KERNEL_TOLERANCE = dict(rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("block_size", [1, 4, 16])
@pytest.mark.parametrize("context_len", [1, 7, 16, 33, 129])
def test_agrees_with_stage_07(nvcc, device, block_size, context_len):
    """The same numbers as the PyTorch version, every layout, every
    length."""
    query, _, _, paged_cache = paged_problem(3, 4, 4, 16, context_len,
                                              block_size, device)
    torch.testing.assert_close(paged_attention_cuda(query, *paged_cache),
                               paged_attention(query, *paged_cache),
                               **KERNEL_TOLERANCE)


def test_agrees_with_the_dense_reference_too(nvcc, device):
    """A second check: not only the same as stage 07, but correct."""
    query, keys, values, paged_cache = paged_problem(4, 8, 8, 64, 100, 16,
                                                      device)
    torch.testing.assert_close(paged_attention_cuda(query, *paged_cache),
                               reference_attention(query, keys, values),
                               rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("num_heads,num_kv_heads", [(8, 2), (16, 8), (4, 4)])
def test_gqa_ratios(nvcc, device, num_heads, num_kv_heads):
    """Query head h must read KV head h // (num_heads // num_kv_heads)."""
    query, _, _, paged_cache = paged_problem(2, num_heads, num_kv_heads, 64,
                                              40, 16, device)
    torch.testing.assert_close(paged_attention_cuda(query, *paged_cache),
                               paged_attention(query, *paged_cache),
                               **KERNEL_TOLERANCE)


def test_ragged_context_lengths(nvcc, device):
    query, _, _, (key_cache, value_cache, block_tables, _) = paged_problem(
        8, 8, 4, 64, 200, 16, device)
    context_lens = torch.tensor([200, 1, 17, 16, 199, 33, 64, 128],
                                dtype=torch.int32, device=device)
    arguments = (query, key_cache, value_cache, block_tables, context_lens)
    torch.testing.assert_close(paged_attention_cuda(*arguments),
                               paged_attention(*arguments), **KERNEL_TOLERANCE)


def test_ignores_junk_beyond_context_len(nvcc, device):
    """The masking check again, because a kernel is much easier to get
    wrong."""
    context_len, block_size = 12, 8
    query, _, _, paged_cache = paged_problem(2, 4, 2, 32, context_len,
                                              block_size, device)
    before = paged_attention_cuda(query, *paged_cache)
    key_cache, value_cache, block_tables, _ = paged_cache
    poison_past_context(key_cache, value_cache, block_tables, context_len,
                        block_size)
    torch.testing.assert_close(before, paged_attention_cuda(query, *paged_cache),
                               **KERNEL_TOLERANCE)


def test_fp16_accumulates_in_fp32(nvcc, device):
    """A long context in fp16 shows careless accumulation.

    An fp16 sum of 2048 fp16 products loses several digits. Accumulate in
    float inside the kernel, and this passes easily.
    """
    query, keys, values, paged_cache = paged_problem(
        2, 8, 8, 128, 2048, 16, device, torch.float16)
    torch.testing.assert_close(paged_attention_cuda(query, *paged_cache),
                               reference_attention(query, keys, values),
                               rtol=3e-3, atol=3e-3)


def test_write_kv_matches_stage_07(nvcc, device):
    """The scatter kernel. The write_kv of stage 07 is the oracle."""
    num_tokens, num_kv_heads, head_dim, block_size, num_blocks = 40, 4, 64, 16, 32
    key = torch.randn(num_tokens, num_kv_heads, head_dim, device=device)
    value = torch.randn(num_tokens, num_kv_heads, head_dim, device=device)
    slots = torch.randperm(num_blocks * block_size,
                           device=device)[:num_tokens].to(torch.int32)

    def caches():
        key_cache = torch.zeros(num_blocks, num_kv_heads, block_size, head_dim,
                                device=device)
        return key_cache, torch.zeros_like(key_cache)

    expected_keys, expected_values = caches()
    write_kv(expected_keys, expected_values, key, value, slots)
    written_keys, written_values = caches()
    write_kv_cuda(written_keys, written_values, key, value, slots)
    torch.testing.assert_close(written_keys, expected_keys)
    torch.testing.assert_close(written_values, expected_values)


def test_the_launch_is_checked(nvcc, device):
    """A kernel that never ran must raise an error, not return garbage.

    Every launch needs C10_CUDA_KERNEL_LAUNCH_CHECK() after it. Without one,
    an illegal configuration fails with no sign, and you debug arithmetic
    that never ran.
    """
    query, _, _, paged_cache = paged_problem(2, 4, 4, 32, 40, 16, device)

    # num_heads must be a multiple of num_kv_heads. A kernel that does not
    # check this reads KV head h // 0, or goes past the end of the cache.
    bad_query = random_query(2, 6, 32, device)
    with pytest.raises(Exception):
        paged_attention_cuda(bad_query, *paged_cache)

    # The good path must still work after it. A sticky CUDA error here means
    # that the failure corrupted the context, and nothing caught it.
    torch.testing.assert_close(paged_attention_cuda(query, *paged_cache),
                               paged_attention(query, *paged_cache),
                               **KERNEL_TOLERANCE)


def test_it_is_actually_faster(nvcc, device):
    """The point of the stage."""
    rows = []
    for num_seqs, context_len in ((8, 256), (32, 512), (64, 1024)):
        query, _, _, paged_cache = paged_problem(
            num_seqs, 16, 8, 128, context_len, 16, device, torch.float16)
        torch_ms = bench_ms(lambda: paged_attention(query, *paged_cache),
                            iters=10)
        cuda_ms = bench_ms(lambda: paged_attention_cuda(query, *paged_cache))
        rows.append((num_seqs, context_len, torch_ms, cuda_ms))

    print(f"\n  {'seqs':>5} {'ctx':>6} {'stage 07':>11} {'stage 08':>11} "
          f"{'speedup':>9}")
    for num_seqs, context_len, torch_ms, cuda_ms in rows:
        print(f"  {num_seqs:>5} {context_len:>6} {torch_ms:>9.2f}ms "
              f"{cuda_ms:>9.3f}ms {torch_ms / cuda_ms:>8.0f}x")

    worst = min(torch_ms / cuda_ms for _, _, torch_ms, cuda_ms in rows)
    assert worst > 3.0, (
        f"only {worst:.1f}x faster in the worst case. Two kernel launches "
        "must be much faster than a Python loop of num_seqs gathers.")
    print(f"\n  \033[1mWorst case: {worst:.0f}x faster than the PyTorch "
          "version.\033[0m")
    print("  \033[2mYou removed num_seqs kernel launches and num_seqs")
    print("  gathers, and the score row never goes to HBM.\033[0m")


def test_how_much_of_the_card_you_are_using(nvcc, device):
    """Not a gate. The number that stage 08b changes.

    The check measures the peak here, and does not read it from a datasheet.
    It measures it next to the kernel, not at import time. A laptop GPU throttles: a peak measured cold
    and a kernel measured hot are a comparison of two different machines.
    """
    import cudalib

    query, _, _, paged_cache = paged_problem(64, 16, 8, 128, 1024, 16,
                                              device, torch.float16)
    context_lens = paged_cache[3]
    peak = cudalib.peak_bandwidth(fresh=True)
    gbs, _ = cudalib.achieved_bandwidth(
        lambda: paged_attention_cuda(query, *paged_cache),
        kv_bytes(context_lens, 8, 128, query.element_size()))

    print(f"\n  peak (measured now)  {peak:>7.0f} GB/s")
    print(f"  your kernel          {gbs:>7.0f} GB/s   "
          f"\033[1m{100 * gbs / peak:.0f}% of it\033[0m")
    print("  \033[2mOne thread for each position walks head_dim alone, so the")
    print("  32 threads of a warp ask for 32 scattered rows at one time. The")
    print("  bytes arrive. The requests are what you spent too much of.")
    print("  Stage 08b.\033[0m")
