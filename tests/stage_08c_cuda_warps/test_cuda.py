"""Stage 08c - warp primitives, occupancy and split-K.

The spec is in app/cuda/s08c_paged_attn_split.cu.

The correctness checks are those of the last two stages, with one more. The
merge across splits must be exact for EVERY number of splits, not only for
the number that your rule selects.

The gate is the throughput at a small batch. A kernel with perfect bandwidth
still loses there, and every interactive request is there.
"""

import pytest
import torch

from app.s07_paged_attn import paged_attention, reference_attention
from app.s08b_cuda_memory import paged_attention_vec
from app.s08c_cuda_warps import paged_attention_split, splits_for
from tests.helpers import (
    kv_bytes,
    paged_problem,
    poison_past_context,
    probe_script,
    report_kernel_stats,
)

KERNEL_TOLERANCE = dict(rtol=1e-3, atol=1e-3)


def _assert_agrees_with_stage_07(query, paged_cache, **split_options):
    torch.testing.assert_close(
        paged_attention_split(query, *paged_cache, **split_options),
        paged_attention(query, *paged_cache), **KERNEL_TOLERANCE)


def _sm_count():
    return torch.cuda.get_device_properties(0).multi_processor_count


@pytest.mark.parametrize("block_size", [1, 4, 16])
@pytest.mark.parametrize("context_len", [1, 7, 16, 33, 129])
def test_agrees_with_stage_07(nvcc, device, block_size, context_len):
    """The oracle still did not move."""
    query, _, _, paged_cache = paged_problem(3, 4, 4, 16, context_len,
                                             block_size, device)
    _assert_agrees_with_stage_07(query, paged_cache)


@pytest.mark.parametrize("splits", [1, 2, 3, 7, 8, 32])
def test_the_merge_is_exact_for_any_split_count(nvcc, device, splits):
    """Forced numbers of splits, also 1, and numbers that do not divide the
    context, so the last split is short or empty.

    exp(max_j - M) is the rescale of the online softmax, one level up. If the
    merge is correct only for even splits, the rescale is wrong, and a
    ragged batch finds it.
    """
    query, keys, values, paged_cache = paged_problem(
        2, 8, 4, 128, 1000, 16, device, torch.float16)
    torch.testing.assert_close(
        paged_attention_split(query, *paged_cache, splits=splits),
        reference_attention(query, keys, values), rtol=3e-3, atol=3e-3)


def test_the_reductions_do_not_race(nvcc, device):
    """Many queries through the same kernel, and all must be correct.

    A block reduction that gives its answer through shared memory needs a
    barrier after every thread READS it, not only before. Without it, the
    first write of the next reduction replaces the result before a slow warp
    reads it. The output is then wrong for a few percent of the elements, on
    some inputs, some of the time. One assertion probably misses that kind
    of bug.

    Split-K makes it worse: more reductions in each block, and a merge pass
    that uses the same scratch again.
    """
    _, _, _, paged_cache = paged_problem(2, 8, 4, 128, 1000, 16, device)
    num_wrong = 0
    for _ in range(25):
        query = torch.randn(2, 8, 128, device=device)
        if not torch.allclose(paged_attention_split(query, *paged_cache,
                                                    splits=32),
                              paged_attention(query, *paged_cache),
                              **KERNEL_TOLERANCE):
            num_wrong += 1
    assert num_wrong == 0, (
        f"{num_wrong} of 25 queries were wrong. Only the numbers of the input "
        "changed between them, so this is a race, not arithmetic. Make sure "
        "that every thread reads a shared-memory result before the next "
        "reduction can overwrite it.")


def test_no_shared_memory_hazards(nvcc, device):
    """The same question as the check above, asked by a tool, not by luck.

    compute-sanitizer instruments every shared-memory access and reports
    each pair that has no barrier between them. It finds the race above also
    when the timing does not show it, and it names both source lines.

    Remember the command. It needs no driver permission, unlike the
    performance counters:

        compute-sanitizer --tool racecheck --racecheck-report analysis \\
            .venv/bin/python your_script.py
    """
    import cudalib

    driver = probe_script(
        "racecheck_08c.py",
        "from app.s08c_cuda_warps import paged_attention_split\n"
        "from tests.helpers import paged_problem\n"
        "query, _, _, paged_cache = paged_problem(2, 8, 4, 128, 512, 16, "
        "'cuda')\n"
        "paged_attention_split(query, *paged_cache, splits=8)\n")
    hazards = cudalib.racecheck(driver)
    if hazards is None:
        pytest.skip("compute-sanitizer is not installed")
    for line in hazards:
        print(f"  {line}")
    assert not hazards, (
        "compute-sanitizer found shared-memory accesses with no barrier "
        "between them. The lines that it names are the two halves of the "
        "race.")


def test_splits_shorter_than_the_context(nvcc, device):
    """More splits than tokens: some blocks get nothing.

    An empty split must report sum = 0 and max = -inf, so that the merge
    ignores it. A maximum of zero poisons the weight of every other split.
    """
    query, _, _, paged_cache = paged_problem(2, 4, 4, 64, 5, 16, device)
    _assert_agrees_with_stage_07(query, paged_cache, splits=16)


@pytest.mark.parametrize("num_heads,num_kv_heads", [(8, 2), (16, 8), (4, 4)])
def test_gqa_ratios(nvcc, device, num_heads, num_kv_heads):
    """Query head h must read KV head h // (num_heads // num_kv_heads)."""
    query, _, _, paged_cache = paged_problem(2, num_heads, num_kv_heads, 64,
                                             40, 16, device)
    _assert_agrees_with_stage_07(query, paged_cache)


def test_ragged_context_lengths(nvcc, device):
    """Every sequence gets the same number of splits, but not the same
    number of tokens in them."""
    query, _, _, (key_cache, value_cache, block_tables, _) = paged_problem(
        8, 8, 4, 64, 200, 16, device)
    context_lens = torch.tensor([200, 1, 17, 16, 199, 33, 64, 128],
                                dtype=torch.int32, device=device)
    _assert_agrees_with_stage_07(query, (key_cache, value_cache, block_tables,
                                         context_lens))


def test_ignores_junk_beyond_context_len(nvcc, device):
    context_len, block_size = 12, 8
    query, _, _, paged_cache = paged_problem(2, 4, 2, 32, context_len,
                                             block_size, device)
    before = paged_attention_split(query, *paged_cache)
    key_cache, value_cache, block_tables, _ = paged_cache
    poison_past_context(key_cache, value_cache, block_tables, context_len,
                        block_size)
    torch.testing.assert_close(before,
                               paged_attention_split(query, *paged_cache),
                               **KERNEL_TOLERANCE)


def test_the_heuristic_actually_splits(nvcc, device):
    """A rule that always returns 1 passes every correctness check above and
    gains nothing. At one sequence there is not enough work to fill the GPU,
    and splits_for must see that."""
    one_seq = splits_for(1, 16, 4096)
    many_seqs = splits_for(64, 16, 1024)
    print(f"\n  {_sm_count()} SMs")
    print(f"  1 seq  x 16 heads, 4096 ctx  ->  {one_seq} splits"
          f"   ({16 * one_seq} blocks)")
    print(f"  64 seqs x 16 heads, 1024 ctx  ->  {many_seqs} splits"
          f"   ({1024 * many_seqs} blocks)")
    assert one_seq > 1, (
        "16 blocks cannot fill this GPU, so a long context at batch 1 must be "
        "split")
    assert one_seq >= many_seqs, "fewer sequences need more splits, not fewer"


def test_it_wins_at_small_batch(nvcc, device):
    """The gate. Where stage 08b has too little parallelism, this does not."""
    import cudalib

    num_heads, context_len = 16, 2048
    rows = []
    for num_seqs in (1, 2, 4):
        query, _, _, paged_cache = paged_problem(
            num_seqs, num_heads, 8, 128, context_len, 16, device,
            torch.float16)
        stage08b_ms, stage08c_ms = cudalib.compare_ms(
            lambda: paged_attention_vec(query, *paged_cache),
            lambda: paged_attention_split(query, *paged_cache))
        rows.append((num_seqs, stage08b_ms, stage08c_ms,
                     splits_for(num_seqs, num_heads, context_len)))

    print(f"\n  {_sm_count()} SMs, {context_len} tokens of context\n")
    print(f"  {'seqs':>5} {'blocks 08b':>11} {'blocks 08c':>11}"
          f" {'stage 08b':>11} {'stage 08c':>11} {'gain':>7}")
    for num_seqs, stage08b_ms, stage08c_ms, num_splits in rows:
        print(f"  {num_seqs:>5} {num_seqs * num_heads:>11} "
              f"{num_seqs * num_heads * num_splits:>11}"
              f" {stage08b_ms:>9.3f}ms {stage08c_ms:>9.3f}ms "
              f"{stage08b_ms / stage08c_ms:>6.2f}x")

    worst = min(row[1] / row[2] for row in rows)
    assert worst > 1.5, (
        f"only {worst:.2f}x over stage 08b at a small batch. Either the warp "
        "reductions did not replace the shared-memory ones, or the grid is "
        "still (num_seqs, num_heads).")
    print(f"\n  \033[1m{worst:.2f}x over stage 08b where the GPU was "
          "empty.\033[0m")


def test_what_split_k_costs_at_a_full_grid(nvcc, device, peak_gbs):
    """Split-K is an exchange, and this is its other side.

    At 64 sequences the grid already covers every SM, so there is no idle
    capacity to use. You still pay for the softmax of each group. Every lane
    of a group now computes its own max, rescale and p. In stage 08b, one
    thread did that for the whole block. Repeated arithmetic costs nothing
    until the kernel stops waiting on memory, and at a full grid it does not
    wait.

    Expect a few percent. If it is worse, the split path runs when it must
    not: `splits_for` must return 1 here.
    """
    import cudalib

    num_seqs, num_heads, context_len = 64, 16, 1024
    query, _, _, paged_cache = paged_problem(num_seqs, num_heads, 8, 128,
                                             context_len, 16, device,
                                             torch.float16)
    num_bytes = kv_bytes(paged_cache[3], 8, 128, query.element_size())
    # Interleave the rounds. Both kernels are at the memory roof here. The
    # difference between them is smaller than the drift of the clock.
    stage08b_ms, stage08c_ms = cudalib.compare_ms(
        lambda: paged_attention_vec(query, *paged_cache),
        lambda: paged_attention_split(query, *paged_cache))

    num_splits = splits_for(num_seqs, num_heads, context_len)
    assert num_splits == 1, (
        f"splits_for gives {num_splits} splits for {num_seqs * num_heads} "
        f"blocks on {_sm_count()} SMs. The grid is already full. A split "
        "only adds a merge.")

    for label, milliseconds in (("stage 08b", stage08b_ms),
                                ("stage 08c", stage08c_ms)):
        gbs = num_bytes / (milliseconds * 1e-3) / 1e9
        print(f"  {label}  {milliseconds:>8.3f}ms  {gbs:>6.0f} GB/s  "
              f"{100 * gbs / peak_gbs:>3.0f}%")
    print(f"  \033[2mthe cost of the split-K shape where it gives nothing: "
          f"{100 * (stage08c_ms / stage08b_ms - 1):.0f}%\033[0m")
    assert stage08b_ms / stage08c_ms > 0.85, (
        f"{stage08b_ms / stage08c_ms:.2f}x. A few percent is the cost of the "
        "softmax of each group, and this is more. Make sure that splits_for "
        "returns 1 here, and that one split does not go through the merge "
        "kernel.")


def test_occupancy_is_reported(nvcc, device):
    """Not a gate. The registers of each thread set how many warps an SM can
    hold, and that sets how much memory latency it can hide."""
    import cudalib

    query, _, _, paged_cache = paged_problem(2, 4, 2, 128, 64, 16, device,
                                             torch.float16)
    paged_attention_split(query, *paged_cache)          # force the build

    stats = cudalib.kernel_stats("s08c_paged_attn_split")
    if not stats:
        pytest.skip("cuobjdump not available")
    spilled = report_kernel_stats(stats)
    occupancy = cudalib.occupancy("s08c_paged_attn_split", "paged_attn_split",
                                  128)
    if occupancy:
        print(f"\n  occupancy from the register count: {100 * occupancy:.0f}% "
              "of the warps that an SM can hold")
    assert not spilled, f"these kernels spill to local memory: {spilled}"


def test_the_counters_agree(nvcc, device):
    """Not a gate, and it skips on most machines.

    test_occupancy_is_reported computes the occupancy from the register
    count, which is a ceiling. This check reads what the SMs really held:
    that ceiling minus what the grid could not supply. At one sequence, the
    gap between the two numbers IS the argument for split-K.

    To read GPU performance counters, a driver option must be on, and it is
    off by default. `./vc ncu 8c` prints the fix. A skip here is normal.
    """
    import cudalib

    driver = probe_script(
        "ncu_probe_08c.py",
        "from app.s08c_cuda_warps import paged_attention_split\n"
        "from tests.helpers import paged_problem\n"
        "query, _, _, paged_cache = paged_problem(1, 16, 8, 128, 4096, 16, "
        "'cuda', torch.float16)\n"
        "paged_attention_split(query, *paged_cache)\n")
    counters = cudalib.ncu_metrics(
        driver,
        ["sm__warps_active.avg.pct_of_peak_sustained_active",
         "dram__bytes.sum.per_second"],
        kernel="regex:paged_attn_split")
    if not counters:
        pytest.skip("the GPU performance counters are not readable here "
                    "(see ./vc ncu 8c)")
    print()
    for name, value in counters.items():
        print(f"  {name.split('.')[0][-44:]:>44}  {value:>14}")
