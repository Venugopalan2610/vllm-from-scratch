"""Stage 08c - warp primitives, occupancy and split-K.

Spec in app/cuda/s08c_paged_attn_split.cu.

The correctness checks are the same as the last two stages, plus one that
matters only here: the merge across splits has to be exact for EVERY split
count, not just the one your heuristic picks.

The gate is small-batch throughput. That is the case a bandwidth-perfect
kernel still loses, and the case every interactive request lives in.
"""

import pytest
import torch

from app.s07_paged_attn import paged_attention, reference_attention
from app.s08b_cuda_memory import paged_attention_vec
from app.s08c_cuda_warps import paged_attention_split, splits_for
from tests.helpers import build_paged, kv_bytes, rand_kv


@pytest.mark.parametrize("block_size", [1, 4, 16])
@pytest.mark.parametrize("L", [1, 7, 16, 33, 129])
def test_agrees_with_stage_07(nvcc, dev, block_size, L):
    """The oracle still has not moved."""
    S, H, KVH, D = 3, 4, 4, 16
    k, v = rand_kv(S, KVH, L, D, dev)
    q = torch.randn(S, H, D, device=dev)
    kc, vc, bt, ctx = build_paged(k, v, block_size)
    torch.testing.assert_close(paged_attention_split(q, kc, vc, bt, ctx),
                               paged_attention(q, kc, vc, bt, ctx),
                               rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("splits", [1, 2, 3, 7, 8, 32])
def test_the_merge_is_exact_for_any_split_count(nvcc, dev, splits):
    """Forced split counts, including 1, including counts that do not divide
    the context and leave the last block short or empty.

    exp(m_j - M) is the same rescale the online softmax already does, one
    level up. If the merge is only right when the splits are even, the
    rescale is wrong and a ragged batch will find it.
    """
    S, H, KVH, D, L = 2, 8, 4, 128, 1000
    k, v = rand_kv(S, KVH, L, D, dev, dtype=torch.float16)
    q = torch.randn(S, H, D, device=dev, dtype=torch.float16)
    kc, vc, bt, ctx = build_paged(k, v, 16)
    torch.testing.assert_close(
        paged_attention_split(q, kc, vc, bt, ctx, splits=splits),
        reference_attention(q, k, v), rtol=3e-3, atol=3e-3)


def test_the_reductions_do_not_race(nvcc, dev):
    """Many queries through the same kernel, all of which must be right.

    A block reduction that hands its answer back through shared memory needs
    a barrier after every thread has READ it, not only before. Without one,
    the next reduction's first write lands on the result a slower warp has
    not collected yet. The output is then wrong for a few percent of the
    elements, on some inputs, some of the time -- which is exactly the shape
    of bug a single assertion is likely to miss.

    Split-K makes it worse: more reductions per block, and a merge pass that
    reuses the same scratch again.
    """
    S, H, KVH, D, L = 2, 8, 4, 128, 1000
    k, v = rand_kv(S, KVH, L, D, dev)
    kc, vc, bt, ctx = build_paged(k, v, 16)

    bad = 0
    for i in range(25):
        q = torch.randn(S, H, D, device=dev)
        want = paged_attention(q, kc, vc, bt, ctx)
        got = paged_attention_split(q, kc, vc, bt, ctx, splits=32)
        if not torch.allclose(got, want, rtol=1e-3, atol=1e-3):
            bad += 1
    assert bad == 0, (
        f"{bad} of 25 queries came back wrong. Nothing about the input "
        "changed between them except the numbers, so this is a race, not "
        "arithmetic. Check that every shared-memory result is read before "
        "the next reduction is allowed to overwrite it.")


def test_no_shared_memory_hazards(nvcc, dev):
    """The same question as the one above, asked by a tool instead of by luck.

    compute-sanitizer instruments every shared-memory access and reports any
    pair that is not separated by a barrier. It finds the race above whether
    or not the timing happens to expose it, and it names both source lines.

    Worth remembering the invocation. It needs no driver permission, unlike
    the performance counters:

        compute-sanitizer --tool racecheck --racecheck-report analysis \
            .venv/bin/python your_script.py
    """
    import cudalib
    from cudalib.build import CACHE

    driver = CACHE / "racecheck_08c.py"
    driver.parent.mkdir(exist_ok=True)
    driver.write_text(
        "import sys, torch\n"
        f"sys.path.insert(0, {str(cudalib.ROOT)!r})\n"
        "from app.s08c_cuda_warps import paged_attention_split as f\n"
        "from tests.helpers import build_paged, rand_kv\n"
        "k, v = rand_kv(2, 4, 512, 128, 'cuda')\n"
        "q = torch.randn(2, 8, 128, device='cuda')\n"
        "kc, vc, bt, ctx = build_paged(k, v, 16)\n"
        "f(q, kc, vc, bt, ctx, splits=8)\n"
        "torch.cuda.synchronize()\n")

    hazards = cudalib.racecheck(driver)
    if hazards is None:
        pytest.skip("compute-sanitizer is not installed")
    for line in hazards:
        print(f"  {line}")
    assert not hazards, (
        "compute-sanitizer found shared-memory accesses with no barrier "
        "between them. The lines it names are the two halves of the race.")


def test_splits_shorter_than_the_context(nvcc, dev):
    """More splits than there are tokens: some blocks get nothing at all.

    An empty split must report l = 0 and m = -inf so the merge ignores it.
    Reporting a zero maximum instead poisons every other split's weight.
    """
    S, H, KVH, D, L = 2, 4, 4, 64, 5
    k, v = rand_kv(S, KVH, L, D, dev)
    q = torch.randn(S, H, D, device=dev)
    kc, vc, bt, ctx = build_paged(k, v, 16)
    torch.testing.assert_close(
        paged_attention_split(q, kc, vc, bt, ctx, splits=16),
        paged_attention(q, kc, vc, bt, ctx), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("H,KVH", [(8, 2), (16, 8), (4, 4)])
def test_gqa_ratios(nvcc, dev, H, KVH):
    """Query head h must read KV head h // (H // KVH)."""
    S, D, L = 2, 64, 40
    k, v = rand_kv(S, KVH, L, D, dev)
    q = torch.randn(S, H, D, device=dev)
    kc, vc, bt, ctx = build_paged(k, v, 16)
    torch.testing.assert_close(paged_attention_split(q, kc, vc, bt, ctx),
                               paged_attention(q, kc, vc, bt, ctx),
                               rtol=1e-3, atol=1e-3)


def test_ragged_context_lengths(nvcc, dev):
    """Every sequence gets the same number of splits, but not the same number
    of tokens in them."""
    S, H, KVH, D, Lmax = 8, 8, 4, 64, 200
    k, v = rand_kv(S, KVH, Lmax, D, dev)
    kc, vc, bt, _ = build_paged(k, v, 16)
    lens = torch.tensor([200, 1, 17, 16, 199, 33, 64, 128],
                        dtype=torch.int32, device=dev)
    q = torch.randn(S, H, D, device=dev)
    torch.testing.assert_close(paged_attention_split(q, kc, vc, bt, lens),
                               paged_attention(q, kc, vc, bt, lens),
                               rtol=1e-3, atol=1e-3)


def test_ignores_junk_beyond_context_len(nvcc, dev):
    S, H, KVH, D, L = 2, 4, 2, 32, 12
    bs = 8
    k, v = rand_kv(S, KVH, L, D, dev)
    kc, vc, bt, _ = build_paged(k, v, bs)
    lens = torch.tensor([L, L], dtype=torch.int32, device=dev)
    q = torch.randn(S, H, D, device=dev)
    before = paged_attention_split(q, kc, vc, bt, lens)

    for s in range(S):
        for b in range(bt.shape[1]):
            phys = int(bt[s, b])
            for off in range(bs):
                if b * bs + off >= L:
                    kc[phys, :, off] = 999.0
                    vc[phys, :, off] = 999.0

    after = paged_attention_split(q, kc, vc, bt, lens)
    torch.testing.assert_close(before, after, rtol=1e-3, atol=1e-3)


def test_the_heuristic_actually_splits(nvcc, dev):
    """A heuristic that always returns 1 passes every correctness check above
    and wins nothing. At one sequence there is not enough work to fill the
    GPU, and splits_for has to notice."""
    p = torch.cuda.get_device_properties(0)
    one = splits_for(1, 16, 4096)
    many = splits_for(64, 16, 1024)
    print(f"\n  {p.multi_processor_count} SMs")
    print(f"  1 seq  x 16 heads, 4096 ctx  ->  {one} splits"
          f"   ({16 * one} blocks)")
    print(f"  64 seqs x 16 heads, 1024 ctx  ->  {many} splits"
          f"   ({1024 * many} blocks)")
    assert one > 1, (
        "16 blocks cannot fill this GPU, so a long context at batch 1 has to "
        "be split")
    assert one >= many, "fewer sequences need more splits, not fewer"


def test_it_wins_at_small_batch(nvcc, dev):
    """The gate. Where stage 08b runs out of parallelism, this does not."""
    import cudalib

    p = torch.cuda.get_device_properties(0)
    H, KVH, D, L = 16, 8, 128, 2048
    rows = []
    for S in (1, 2, 4):
        k, v = rand_kv(S, KVH, L, D, dev, dtype=torch.float16)
        q = torch.randn(S, H, D, device=dev, dtype=torch.float16)
        kc, vc, bt, ctx = build_paged(k, v, 16)
        a = cudalib.bench_ms(lambda: paged_attention_vec(q, kc, vc, bt, ctx),
                             best_of=2)
        b = cudalib.bench_ms(lambda: paged_attention_split(q, kc, vc, bt, ctx),
                             best_of=2)
        rows.append((S, a, b, splits_for(S, H, L)))

    print(f"\n  {p.multi_processor_count} SMs, {L} tokens of context\n")
    print(f"  {'seqs':>5} {'blocks 08b':>11} {'blocks 08c':>11}"
          f" {'stage 08b':>11} {'stage 08c':>11} {'gain':>7}")
    for S, a, b, sp in rows:
        print(f"  {S:>5} {S * H:>11} {S * H * sp:>11}"
              f" {a:>9.3f}ms {b:>9.3f}ms {a / b:>6.2f}x")

    worst = min(a / b for _, a, b, _ in rows)
    assert worst > 1.5, (
        f"only {worst:.2f}x over stage 08b at small batch. Either the warp "
        "reductions did not replace the shared-memory ones, or the grid is "
        "still (num_seqs, num_heads).")
    print(f"\n  \033[1m{worst:.2f}x over stage 08b where the GPU was empty."
          f"\033[0m")


def test_what_split_k_costs_at_a_full_grid(nvcc, dev):
    """Split-K is a trade, and this is the other side of it.

    At 64 sequences the grid already covered every SM, so there was no
    idleness to sell. What you still pay for is the per-group softmax: every
    lane of a group now works out its own m, alpha and p, where stage 08b had
    one thread do it for the whole block. Redundant arithmetic is free right
    up to the point where the kernel is not waiting on memory any more, and
    at a full grid it is not.

    Expect a few percent. If it is worse than that, the split path is being
    taken when it should not be: `splits_for` has to return 1 here.
    """
    import cudalib

    S, H, KVH, D, L = 64, 16, 8, 128, 1024
    k, v = rand_kv(S, KVH, L, D, dev, dtype=torch.float16)
    q = torch.randn(S, H, D, device=dev, dtype=torch.float16)
    kc, vc, bt, ctx = build_paged(k, v, 16)
    by = kv_bytes(ctx, KVH, D, q.element_size())

    peak = cudalib.peak_bandwidth(fresh=True)
    # best_of, because both kernels sit at the memory roof here and the
    # difference between them is smaller than the drift from one throttling
    # down while the other is measured.
    a = cudalib.bench_ms(lambda: paged_attention_vec(q, kc, vc, bt, ctx),
                         best_of=3)
    b = cudalib.bench_ms(lambda: paged_attention_split(q, kc, vc, bt, ctx),
                         best_of=3)
    ga, gb = by / (a * 1e-3) / 1e9, by / (b * 1e-3) / 1e9

    assert splits_for(S, H, L) == 1, (
        f"splits_for says {splits_for(S, H, L)} splits for {S * H} blocks on "
        f"{torch.cuda.get_device_properties(0).multi_processor_count} SMs. "
        "The grid is already full; splitting it only adds a merge.")

    print(f"\n  peak {peak:.0f} GB/s, measured now")
    print(f"  stage 08b  {a:>8.3f}ms  {ga:>6.0f} GB/s  {100 * ga / peak:>3.0f}%")
    print(f"  stage 08c  {b:>8.3f}ms  {gb:>6.0f} GB/s  {100 * gb / peak:>3.0f}%")
    print(f"  \033[2mcost of the split-K shape where it buys nothing: "
          f"{100 * (b / a - 1):.0f}%\033[0m")
    assert a / b > 0.85, (
        f"{a / b:.2f}x. A few percent is the price of the per-group softmax; "
        "this is more than that. Check that splits_for returns 1 here, and "
        "that one split does not go through the merge kernel at all.")


def test_occupancy_is_reported(nvcc, dev):
    """Not a gate. Registers per thread decide how many warps an SM can hold,
    and that is how much memory latency it can hide."""
    import cudalib

    S, H, KVH, D, L = 2, 4, 2, 128, 64
    k, v = rand_kv(S, KVH, L, D, dev, dtype=torch.float16)
    q = torch.randn(S, H, D, device=dev, dtype=torch.float16)
    kc, vc, bt, ctx = build_paged(k, v, 16)
    paged_attention_split(q, kc, vc, bt, ctx)

    stats = cudalib.kernel_stats("s08c_paged_attn_split")
    if not stats:
        pytest.skip("cuobjdump not available")
    print()
    for name, s in sorted(stats.items()):
        if "EmptyKernel" in name:
            continue
        print(f"  {name[-40:]:>40}  {s['reg']:>3} regs  "
              f"{s['shared']:>5}B shared  {s['local']:>4}B spilled")
    occ = cudalib.occupancy("s08c_paged_attn_split", "paged_attn_split", 128)
    if occ:
        print(f"\n  occupancy from the register count: {100 * occ:.0f}% "
              f"of the warps an SM can hold")
    spilled = {n: s for n, s in stats.items() if s.get("local")}
    assert not spilled, f"these kernels spill to local memory: {list(spilled)}"


def test_the_counters_agree(nvcc, dev):
    """Not a gate, and it skips on most machines.

    test_occupancy_is_reported computes occupancy from the register count,
    which is a ceiling. This reads what the SMs actually held, which is that
    ceiling minus whatever the grid could not supply. At one sequence, the
    gap between the two numbers IS the argument for split-K.

    Reading GPU performance counters needs a driver option that is off by
    default. `./vc ncu 8c` prints the fix. Skipping here is normal.
    """
    import cudalib
    from cudalib.build import CACHE

    driver = CACHE / "ncu_probe_08c.py"
    driver.parent.mkdir(exist_ok=True)
    driver.write_text(
        "import sys, torch\n"
        f"sys.path.insert(0, {str(cudalib.ROOT)!r})\n"
        "from app.s08c_cuda_warps import paged_attention_split as f\n"
        "from tests.helpers import build_paged, rand_kv\n"
        "k, v = rand_kv(1, 8, 4096, 128, 'cuda', dtype=torch.float16)\n"
        "q = torch.randn(1, 16, 128, device='cuda', dtype=torch.float16)\n"
        "kc, vc, bt, ctx = build_paged(k, v, 16)\n"
        "f(q, kc, vc, bt, ctx)\n"
        "torch.cuda.synchronize()\n")

    got = cudalib.ncu_metrics(
        driver,
        ["sm__warps_active.avg.pct_of_peak_sustained_active",
         "dram__bytes.sum.per_second"],
        kernel="regex:paged_attn_split")
    if not got:
        pytest.skip("GPU performance counters are not readable here "
                    "(see ./vc ncu 8c)")
    print()
    for name, value in got.items():
        print(f"  {name.split('.')[0][-44:]:>44}  {value:>14}")
