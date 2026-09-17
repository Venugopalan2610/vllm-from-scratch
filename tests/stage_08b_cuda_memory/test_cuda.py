"""Stage 08b - coalescing and vectorization.

Spec in app/cuda/s08b_paged_attn_vec.cu.

Correctness is checked against stage 07, exactly as stage 08 was: the
arithmetic did not change, so the answers must not either. What changed is
the bytes per transaction, and the last two checks are about that.
"""

import pytest
import torch

from app.s07_paged_attn import paged_attention, reference_attention
from app.s08_paged_cuda import paged_attention_cuda
from app.s08b_cuda_memory import paged_attention_vec
from tests.helpers import build_paged, kv_bytes, rand_kv


@pytest.mark.parametrize("block_size", [1, 4, 16])
@pytest.mark.parametrize("L", [1, 7, 16, 33, 129])
def test_agrees_with_stage_07(nvcc, dev, block_size, L):
    """The oracle has not moved."""
    S, H, KVH, D = 3, 4, 4, 16
    k, v = rand_kv(S, KVH, L, D, dev)
    q = torch.randn(S, H, D, device=dev)
    kc, vc, bt, ctx = build_paged(k, v, block_size)
    torch.testing.assert_close(paged_attention_vec(q, kc, vc, bt, ctx),
                               paged_attention(q, kc, vc, bt, ctx),
                               rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("D", [16, 32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_every_head_dim_and_dtype(nvcc, dev, D, dtype):
    """VEC is 4 floats or 8 halves, so head_dim and dtype together decide how
    many threads share a row. Every combination has to land on the same
    answer, including the ones where a row is narrower than a warp."""
    S, H, KVH, L = 2, 4, 2, 70
    k, v = rand_kv(S, KVH, L, D, dev, dtype=dtype)
    q = torch.randn(S, H, D, device=dev, dtype=dtype)
    kc, vc, bt, ctx = build_paged(k, v, 16)
    tol = 1e-3 if dtype == torch.float32 else 3e-3
    torch.testing.assert_close(paged_attention_vec(q, kc, vc, bt, ctx),
                               paged_attention(q, kc, vc, bt, ctx),
                               rtol=tol, atol=tol)


@pytest.mark.parametrize("H,KVH", [(8, 2), (16, 8), (4, 4)])
def test_gqa_ratios(nvcc, dev, H, KVH):
    """Query head h must read KV head h // (H // KVH)."""
    S, D, L = 2, 64, 40
    k, v = rand_kv(S, KVH, L, D, dev)
    q = torch.randn(S, H, D, device=dev)
    kc, vc, bt, ctx = build_paged(k, v, 16)
    torch.testing.assert_close(paged_attention_vec(q, kc, vc, bt, ctx),
                               paged_attention(q, kc, vc, bt, ctx),
                               rtol=1e-3, atol=1e-3)


def test_ragged_context_lengths(nvcc, dev):
    """Rows in flight at the same time now finish at different times."""
    S, H, KVH, D, Lmax = 8, 8, 4, 64, 200
    k, v = rand_kv(S, KVH, Lmax, D, dev)
    kc, vc, bt, _ = build_paged(k, v, 16)
    lens = torch.tensor([200, 1, 17, 16, 199, 33, 64, 128],
                        dtype=torch.int32, device=dev)
    q = torch.randn(S, H, D, device=dev)
    torch.testing.assert_close(paged_attention_vec(q, kc, vc, bt, lens),
                               paged_attention(q, kc, vc, bt, lens),
                               rtol=1e-3, atol=1e-3)


def test_ignores_junk_beyond_context_len(nvcc, dev):
    """A wide load reads 16 bytes whether you wanted them all or not, so the
    mask has to survive vectorization."""
    S, H, KVH, D, L = 2, 4, 2, 32, 12
    bs = 8
    k, v = rand_kv(S, KVH, L, D, dev)
    kc, vc, bt, _ = build_paged(k, v, bs)
    lens = torch.tensor([L, L], dtype=torch.int32, device=dev)
    q = torch.randn(S, H, D, device=dev)
    before = paged_attention_vec(q, kc, vc, bt, lens)

    for s in range(S):
        for b in range(bt.shape[1]):
            phys = int(bt[s, b])
            for off in range(bs):
                if b * bs + off >= L:
                    kc[phys, :, off] = 999.0
                    vc[phys, :, off] = 999.0

    after = paged_attention_vec(q, kc, vc, bt, lens)
    torch.testing.assert_close(before, after, rtol=1e-3, atol=1e-3)


def test_fp16_accumulates_in_fp32(nvcc, dev):
    """2048 half products, summed in half, lose several digits."""
    S, H, KVH, D, L = 2, 8, 8, 128, 2048
    k, v = rand_kv(S, KVH, L, D, dev, dtype=torch.float16)
    q = torch.randn(S, H, D, device=dev, dtype=torch.float16)
    kc, vc, bt, ctx = build_paged(k, v, 16)
    torch.testing.assert_close(paged_attention_vec(q, kc, vc, bt, ctx),
                               reference_attention(q, k, v),
                               rtol=3e-3, atol=3e-3)


def test_no_register_spills(nvcc, dev):
    """Spilled registers are DRAM wearing a register's name.

    ptxas allocates registers per thread. Ask for more than exist and it puts
    the overflow in local memory, which is global memory, which is the thing
    this whole stage is trying to touch less.
    """
    import cudalib

    S, H, KVH, D, L = 2, 4, 2, 128, 64
    k, v = rand_kv(S, KVH, L, D, dev, dtype=torch.float16)
    q = torch.randn(S, H, D, device=dev, dtype=torch.float16)
    kc, vc, bt, ctx = build_paged(k, v, 16)
    paged_attention_vec(q, kc, vc, bt, ctx)          # force the build

    stats = cudalib.kernel_stats("s08b_paged_attn_vec")
    if not stats:
        pytest.skip("cuobjdump not available")
    print()
    for name, s in sorted(stats.items()):
        if "EmptyKernel" in name:
            continue
        print(f"  {name[-40:]:>40}  {s['reg']:>3} regs  "
              f"{s['shared']:>5}B shared  {s['local']:>4}B spilled")
    spilled = {n: s for n, s in stats.items() if s.get("local")}
    assert not spilled, f"these kernels spill to local memory: {list(spilled)}"


def test_it_moves_more_bytes_per_second(nvcc, dev):
    """The gate. Same arithmetic, same answers, more bandwidth.

    The ratio against stage 08 is the honest measurement: both kernels are
    timed in the same thermal state, seconds apart. The percentage of peak is
    printed next to it and gated loosely, because how close a card gets to
    its own streaming bandwidth depends on the card.
    """
    import cudalib

    peak = cudalib.peak_bandwidth(fresh=True)
    rows = []
    for S, L in ((8, 256), (32, 512), (64, 1024)):
        H, KVH, D = 16, 8, 128
        k, v = rand_kv(S, KVH, L, D, dev, dtype=torch.float16)
        q = torch.randn(S, H, D, device=dev, dtype=torch.float16)
        kc, vc, bt, ctx = build_paged(k, v, 16)
        by = kv_bytes(ctx, KVH, D, q.element_size())

        a = cudalib.bench_ms(lambda: paged_attention_cuda(q, kc, vc, bt, ctx),
                             best_of=2)
        b = cudalib.bench_ms(lambda: paged_attention_vec(q, kc, vc, bt, ctx),
                             best_of=2)
        rows.append((S, L, a, b, by / (a * 1e-3) / 1e9, by / (b * 1e-3) / 1e9))

    print(f"\n  measured peak: {peak:.0f} GB/s\n")
    print(f"  {'seqs':>5} {'ctx':>6} | {'stage 08':>9} {'GB/s':>7} {'%pk':>5}"
          f" | {'stage 08b':>10} {'GB/s':>7} {'%pk':>5} | {'gain':>6}")
    for S, L, a, b, ga, gb in rows:
        print(f"  {S:>5} {L:>6} | {a:>8.3f}ms {ga:>7.0f} {100 * ga / peak:>4.0f}%"
              f" | {b:>9.3f}ms {gb:>7.0f} {100 * gb / peak:>4.0f}%"
              f" | {a / b:>5.2f}x")

    worst = min(a / b for _, _, a, b, _, _ in rows)
    assert worst > 1.5, (
        f"only {worst:.2f}x faster than stage 08. Check that consecutive "
        "threads read consecutive addresses, and that each one moves 16 bytes."
    )

    big = rows[-1]
    frac = big[5] / peak
    assert frac > 0.40, (
        f"only {100 * frac:.0f}% of measured peak bandwidth at {big[0]} "
        f"sequences. A decode attention kernel at this size should be close "
        "to streaming.")
    print(f"\n  \033[1m{worst:.2f}x over stage 08, at {100 * frac:.0f}% "
          f"of what this card can stream.\033[0m")
    print("  \033[2mThe arithmetic is identical. You only changed which thread")
    print("  touches which byte.\033[0m")


def test_the_counters_agree(nvcc, dev):
    """Not a gate, and it skips on most machines.

    Everything else in this file infers coalescing from a stopwatch. This
    reads it: the fraction of every 32-byte sector fetched that the kernel
    actually used. Stage 08 leaves most of each sector on the floor.

    Reading GPU performance counters needs a driver option that is off by
    default. `./vc ncu 8b` prints the fix. Skipping here is normal.
    """
    import cudalib
    from cudalib.build import CACHE

    driver = CACHE / "ncu_probe_08b.py"
    driver.parent.mkdir(exist_ok=True)
    driver.write_text(
        "import sys, torch\n"
        f"sys.path.insert(0, {str(cudalib.ROOT)!r})\n"
        "from app.s08b_cuda_memory import paged_attention_vec as f\n"
        "from tests.helpers import build_paged, rand_kv\n"
        "k, v = rand_kv(32, 8, 1024, 128, 'cuda', dtype=torch.float16)\n"
        "q = torch.randn(32, 16, 128, device='cuda', dtype=torch.float16)\n"
        "kc, vc, bt, ctx = build_paged(k, v, 16)\n"
        "f(q, kc, vc, bt, ctx)\n"
        "torch.cuda.synchronize()\n")

    got = cudalib.ncu_metrics(
        driver,
        ["smsp__average_data_bytes_per_sector_mem_global_op_ld.pct",
         "dram__bytes.sum.per_second"],
        kernel="regex:paged_attn_vec")
    if not got:
        pytest.skip("GPU performance counters are not readable here "
                    "(see ./vc ncu 8b)")
    print()
    for name, value in got.items():
        print(f"  {name.split('.')[0][-44:]:>44}  {value:>14}")
