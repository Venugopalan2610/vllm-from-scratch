"""Stage 08b - coalescing and vectorization.

The spec is in app/cuda/s08b_paged_attn_vec.cu.

These checks compare with stage 07, as stage 08 did. The arithmetic did not
change, so the answers must not change.

The bytes in each transaction changed. The last two checks examine that.
"""

import pytest
import torch

from app.s07_paged_attn import paged_attention, reference_attention
from app.s08_paged_cuda import paged_attention_cuda
from app.s08b_cuda_memory import paged_attention_vec
from tests.helpers import (
    kv_bytes,
    paged_problem,
    poison_past_context,
    probe_script,
    report_kernel_stats,
)

KERNEL_TOLERANCE = dict(rtol=1e-3, atol=1e-3)


def _assert_agrees_with_stage_07(query, paged_cache, tolerance=1e-3):
    torch.testing.assert_close(paged_attention_vec(query, *paged_cache),
                               paged_attention(query, *paged_cache),
                               rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("block_size", [1, 4, 16])
@pytest.mark.parametrize("context_len", [1, 7, 16, 33, 129])
def test_agrees_with_stage_07(nvcc, device, block_size, context_len):
    """The oracle did not move."""
    query, _, _, paged_cache = paged_problem(3, 4, 4, 16, context_len,
                                             block_size, device)
    _assert_agrees_with_stage_07(query, paged_cache)


@pytest.mark.parametrize("head_dim", [16, 32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_every_head_dim_and_dtype(nvcc, device, head_dim, dtype):
    """VEC is 4 floats or 8 halves, so head_dim and the dtype together set
    the number of threads that share a row. Every combination must give the
    same answer, also when a row is narrower than a warp."""
    query, _, _, paged_cache = paged_problem(2, 4, 2, head_dim, 70, 16, device,
                                             dtype)
    _assert_agrees_with_stage_07(query, paged_cache,
                                 1e-3 if dtype == torch.float32 else 3e-3)


@pytest.mark.parametrize("num_heads,num_kv_heads", [(8, 2), (16, 8), (4, 4)])
def test_gqa_ratios(nvcc, device, num_heads, num_kv_heads):
    """Query head h must read KV head h // (num_heads // num_kv_heads)."""
    query, _, _, paged_cache = paged_problem(2, num_heads, num_kv_heads, 64,
                                             40, 16, device)
    _assert_agrees_with_stage_07(query, paged_cache)


def test_ragged_context_lengths(nvcc, device):
    """Rows in flight at the same time now finish at different times."""
    query, _, _, (key_cache, value_cache, block_tables, _) = paged_problem(
        8, 8, 4, 64, 200, 16, device)
    context_lens = torch.tensor([200, 1, 17, 16, 199, 33, 64, 128],
                                dtype=torch.int32, device=device)
    _assert_agrees_with_stage_07(query, (key_cache, value_cache, block_tables,
                                         context_lens))


def test_ignores_junk_beyond_context_len(nvcc, device):
    """A wide load reads 16 bytes, all of them, so the mask must work with
    vectors too."""
    context_len, block_size = 12, 8
    query, _, _, paged_cache = paged_problem(2, 4, 2, 32, context_len,
                                             block_size, device)
    before = paged_attention_vec(query, *paged_cache)
    key_cache, value_cache, block_tables, _ = paged_cache
    poison_past_context(key_cache, value_cache, block_tables, context_len,
                        block_size)
    torch.testing.assert_close(before, paged_attention_vec(query, *paged_cache),
                               **KERNEL_TOLERANCE)


def test_fp16_accumulates_in_fp32(nvcc, device):
    """A half sum of 2048 half products loses several digits."""
    query, keys, values, paged_cache = paged_problem(
        2, 8, 8, 128, 2048, 16, device, torch.float16)
    torch.testing.assert_close(paged_attention_vec(query, *paged_cache),
                               reference_attention(query, keys, values),
                               rtol=3e-3, atol=3e-3)


def test_no_register_spills(nvcc, device):
    """A spilled register is DRAM with the name of a register.

    ptxas allocates registers for each thread. Ask for more than exist, and
    it puts the rest in local memory. Local memory is global memory, and
    this stage tries to touch global memory less.
    """
    import cudalib

    query, _, _, paged_cache = paged_problem(2, 4, 2, 128, 64, 16, device,
                                             torch.float16)
    paged_attention_vec(query, *paged_cache)          # force the build

    stats = cudalib.kernel_stats("s08b_paged_attn_vec")
    if not stats:
        pytest.skip("cuobjdump not available")
    spilled = report_kernel_stats(stats)
    assert not spilled, f"these kernels spill to local memory: {spilled}"


def test_it_moves_more_bytes_per_second(nvcc, device, peak_gbs):
    """The gate. The same arithmetic, the same answers, more bandwidth.

    The ratio against stage 08 is the true measurement: compare_ms
    interleaves the rounds of the two kernels, so both see the same thermal
    state. The check prints the fraction of the peak next to it, with a loose
    gate. How near a card gets to its own streaming bandwidth depends on the
    card.
    """
    import cudalib

    rows = []
    for num_seqs, context_len in ((8, 256), (32, 512), (64, 1024)):
        query, _, _, paged_cache = paged_problem(num_seqs, 16, 8, 128,
                                                 context_len, 16, device,
                                                 torch.float16)
        num_bytes = kv_bytes(paged_cache[3], 8, 128, query.element_size())
        stage08_ms, stage08b_ms = cudalib.compare_ms(
            lambda: paged_attention_cuda(query, *paged_cache),
            lambda: paged_attention_vec(query, *paged_cache))
        rows.append((num_seqs, context_len, stage08_ms, stage08b_ms,
                     num_bytes / (stage08_ms * 1e-3) / 1e9,
                     num_bytes / (stage08b_ms * 1e-3) / 1e9))

    print(f"\n  measured peak: {peak_gbs:.0f} GB/s\n")
    print(f"  {'seqs':>5} {'ctx':>6} | {'stage 08':>9} {'GB/s':>7} {'%pk':>5}"
          f" | {'stage 08b':>10} {'GB/s':>7} {'%pk':>5} | {'gain':>6}")
    for (num_seqs, context_len, stage08_ms, stage08b_ms, stage08_gbs,
         stage08b_gbs) in rows:
        print(f"  {num_seqs:>5} {context_len:>6} | {stage08_ms:>8.3f}ms "
              f"{stage08_gbs:>7.0f} {100 * stage08_gbs / peak_gbs:>4.0f}%"
              f" | {stage08b_ms:>9.3f}ms {stage08b_gbs:>7.0f} "
              f"{100 * stage08b_gbs / peak_gbs:>4.0f}%"
              f" | {stage08_ms / stage08b_ms:>5.2f}x")

    worst = min(row[2] / row[3] for row in rows)
    assert worst > 1.5, (
        f"only {worst:.2f}x faster than stage 08. Make sure that adjacent "
        "threads read adjacent addresses, and that each one moves 16 bytes.")
    largest = rows[-1]
    fraction = largest[5] / peak_gbs
    assert fraction > 0.40, (
        f"only {100 * fraction:.0f}% of the measured peak bandwidth at "
        f"{largest[0]} sequences. A decode attention kernel at this size must "
        "be near streaming.")
    print(f"\n  \033[1m{worst:.2f}x over stage 08, at {100 * fraction:.0f}% "
          "of what this card can stream.\033[0m")
    print("  \033[2mThe arithmetic is the same. You only changed which thread")
    print("  touches which byte.\033[0m")


def test_the_counters_agree(nvcc, device):
    """Not a gate, and it skips on most machines.

    Every other check here infers coalescing from a stopwatch. This check
    reads it: the fraction of each 32-byte sector fetched that the kernel
    used. Stage 08 wastes most of each sector.

    To read GPU performance counters, a driver option must be on, and it is
    off by default. `./vc ncu 8b` prints the fix. A skip here is normal.
    """
    import cudalib

    driver = probe_script(
        "ncu_probe_08b.py",
        "from app.s08b_cuda_memory import paged_attention_vec\n"
        "from tests.helpers import paged_problem\n"
        "query, _, _, paged_cache = paged_problem(32, 16, 8, 128, 1024, 16, "
        "'cuda', torch.float16)\n"
        "paged_attention_vec(query, *paged_cache)\n")

    counters = cudalib.ncu_metrics(
        driver,
        ["smsp__average_data_bytes_per_sector_mem_global_op_ld.pct",
         "dram__bytes.sum.per_second"],
        kernel="regex:paged_attn_vec")
    if not counters:
        pytest.skip("the GPU performance counters are not readable here "
                    "(see ./vc ncu 8b)")
    print()
    for name, value in counters.items():
        print(f"  {name.split('.')[0][-44:]:>44}  {value:>14}")
