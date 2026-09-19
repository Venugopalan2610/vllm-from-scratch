"""Stage 08 - Paged attention in CUDA.

Spec in app/cuda/s08_paged_attn.cu and app/s08_paged_cuda.py.

Every correctness check here compares against stage 07, which is your oracle.
The last two checks matter most. The kernel must be faster than the PyTorch
version. And the bandwidth that it reports is the number that stage 08b moves.
"""

import pytest
import torch

from app.s07_paged_attn import paged_attention, reference_attention, write_kv
from app.s08_paged_cuda import paged_attention_cuda, write_kv_cuda
from tests.helpers import bench_ms, build_paged, kv_bytes, rand_kv


@pytest.mark.parametrize("block_size", [1, 4, 16])
@pytest.mark.parametrize("L", [1, 7, 16, 33, 129])
def test_agrees_with_stage_07(nvcc, dev, block_size, L):
    """Same numbers as the PyTorch version, every layout, every length."""
    S, H, KVH, D = 3, 4, 4, 16
    k, v = rand_kv(S, KVH, L, D, dev)
    q = torch.randn(S, H, D, device=dev)
    kc, vc, bt, ctx = build_paged(k, v, block_size)

    want = paged_attention(q, kc, vc, bt, ctx)
    got = paged_attention_cuda(q, kc, vc, bt, ctx)
    torch.testing.assert_close(got, want, rtol=1e-3, atol=1e-3)


def test_agrees_with_the_dense_reference_too(nvcc, dev):
    """Belt and braces: not just consistent with stage 07, but correct."""
    S, H, KVH, D, L = 4, 8, 8, 64, 100
    k, v = rand_kv(S, KVH, L, D, dev)
    q = torch.randn(S, H, D, device=dev)
    kc, vc, bt, ctx = build_paged(k, v, 16)
    want = reference_attention(q, k, v)
    got = paged_attention_cuda(q, kc, vc, bt, ctx)
    torch.testing.assert_close(got, want, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("H,KVH", [(8, 2), (16, 8), (4, 4)])
def test_gqa_ratios(nvcc, dev, H, KVH):
    """Query head h must read KV head h // (H // KVH)."""
    S, D, L = 2, 64, 40
    k, v = rand_kv(S, KVH, L, D, dev)
    q = torch.randn(S, H, D, device=dev)
    kc, vc, bt, ctx = build_paged(k, v, 16)
    want = paged_attention(q, kc, vc, bt, ctx)
    got = paged_attention_cuda(q, kc, vc, bt, ctx)
    torch.testing.assert_close(got, want, rtol=1e-3, atol=1e-3)


def test_ragged_context_lengths(nvcc, dev):
    S, H, KVH, D, Lmax = 8, 8, 4, 64, 200
    k, v = rand_kv(S, KVH, Lmax, D, dev)
    kc, vc, bt, _ = build_paged(k, v, 16)
    lens = torch.tensor([200, 1, 17, 16, 199, 33, 64, 128],
                        dtype=torch.int32, device=dev)
    q = torch.randn(S, H, D, device=dev)
    want = paged_attention(q, kc, vc, bt, lens)
    got = paged_attention_cuda(q, kc, vc, bt, lens)
    torch.testing.assert_close(got, want, rtol=1e-3, atol=1e-3)


def test_ignores_junk_beyond_context_len(nvcc, dev):
    """The masking test again, because a kernel is much easier to get wrong."""
    S, H, KVH, D, L = 2, 4, 2, 32, 12
    bs = 8
    k, v = rand_kv(S, KVH, L, D, dev)
    kc, vc, bt, _ = build_paged(k, v, bs)
    lens = torch.tensor([L, L], dtype=torch.int32, device=dev)
    q = torch.randn(S, H, D, device=dev)
    before = paged_attention_cuda(q, kc, vc, bt, lens)

    for s in range(S):
        for b in range(bt.shape[1]):
            phys = int(bt[s, b])
            for off in range(bs):
                if b * bs + off >= L:
                    kc[phys, :, off] = 999.0
                    vc[phys, :, off] = 999.0

    after = paged_attention_cuda(q, kc, vc, bt, lens)
    torch.testing.assert_close(before, after, rtol=1e-3, atol=1e-3)


def test_fp16_accumulates_in_fp32(nvcc, dev):
    """Long contexts in fp16 are where sloppy accumulation shows up.

    Sum 2048 fp16 products in fp16 and you lose several digits. Accumulate in
    float inside the kernel and this passes comfortably.
    """
    S, H, KVH, D, L = 2, 8, 8, 128, 2048
    k, v = rand_kv(S, KVH, L, D, dev, dtype=torch.float16)
    q = torch.randn(S, H, D, device=dev, dtype=torch.float16)
    kc, vc, bt, ctx = build_paged(k, v, 16)
    want = reference_attention(q, k, v)
    got = paged_attention_cuda(q, kc, vc, bt, ctx)
    torch.testing.assert_close(got, want, rtol=3e-3, atol=3e-3)


def test_write_kv_matches_stage_07(nvcc, dev):
    """The scatter kernel. Stage 07's write_kv is the oracle."""
    T, KVH, D, BS, NB = 40, 4, 64, 16, 32
    key = torch.randn(T, KVH, D, device=dev)
    val = torch.randn(T, KVH, D, device=dev)
    slots = torch.randperm(NB * BS, device=dev)[:T].to(torch.int32)

    want_k = torch.zeros(NB, KVH, BS, D, device=dev)
    want_v = torch.zeros_like(want_k)
    write_kv(want_k, want_v, key, val, slots)

    got_k = torch.zeros_like(want_k)
    got_v = torch.zeros_like(want_k)
    write_kv_cuda(got_k, got_v, key, val, slots)

    torch.testing.assert_close(got_k, want_k)
    torch.testing.assert_close(got_v, want_v)


def test_the_launch_is_checked(nvcc, dev):
    """A kernel that never ran must raise, not return silent garbage.

    Every launch needs C10_CUDA_KERNEL_LAUNCH_CHECK() after it. Without one,
    an illegal configuration fails invisibly and you spend an evening
    debugging arithmetic that never executed.
    """
    S, H, KVH, D, L = 2, 4, 4, 32, 40
    k, v = rand_kv(S, KVH, L, D, dev)
    q = torch.randn(S, H, D, device=dev)
    kc, vc, bt, ctx = build_paged(k, v, 16)

    # num_heads must be a multiple of num_kv_heads. A kernel that does not
    # check this reads KV head h // 0 or walks off the cache.
    bad = torch.randn(S, 6, D, device=dev)
    with pytest.raises(Exception):
        paged_attention_cuda(bad, kc, vc, bt, ctx)

    # ... and the good path still works afterwards. A sticky CUDA error here
    # means the failure corrupted the context rather than being caught.
    torch.testing.assert_close(paged_attention_cuda(q, kc, vc, bt, ctx),
                               paged_attention(q, kc, vc, bt, ctx),
                               rtol=1e-3, atol=1e-3)


def test_it_is_actually_faster(nvcc, dev):
    """The whole point of the stage."""
    rows = []
    for S, L in ((8, 256), (32, 512), (64, 1024)):
        H, KVH, D = 16, 8, 128
        k, v = rand_kv(S, KVH, L, D, dev, dtype=torch.float16)
        q = torch.randn(S, H, D, device=dev, dtype=torch.float16)
        kc, vc, bt, ctx = build_paged(k, v, 16)

        torch_ms = bench_ms(lambda: paged_attention(q, kc, vc, bt, ctx), iters=10)
        cuda_ms = bench_ms(lambda: paged_attention_cuda(q, kc, vc, bt, ctx))
        rows.append((S, L, torch_ms, cuda_ms))

    print(f"\n  {'seqs':>5} {'ctx':>6} {'stage 07':>11} {'stage 08':>11} {'speedup':>9}")
    for S, L, a, b in rows:
        print(f"  {S:>5} {L:>6} {a:>9.2f}ms {b:>9.3f}ms {a / b:>8.0f}x")

    worst = min(a / b for _, _, a, b in rows)
    assert worst > 3.0, (
        f"only {worst:.1f}x faster at best. Two kernel launches should crush a "
        "Python loop that does num_seqs separate gathers."
    )
    print(f"\n  \033[1mWorst case: {worst:.0f}x faster than the PyTorch version.\033[0m")
    print("  \033[2mYou deleted num_seqs kernel launches and num_seqs gathers,")
    print("  and the score row never reaches HBM at all.\033[0m")


def test_how_much_of_the_card_you_are_using(nvcc, dev):
    """Not a gate. The number stage 08b exists to move.

    Peak is measured right here rather than read from a datasheet, and right
    next to the kernel rather than at import time, because a laptop GPU
    throttles: a peak measured cold and a kernel measured hot is a comparison
    between two different machines.
    """
    import cudalib

    S, H, KVH, D, L = 64, 16, 8, 128, 1024
    k, v = rand_kv(S, KVH, L, D, dev, dtype=torch.float16)
    q = torch.randn(S, H, D, device=dev, dtype=torch.float16)
    kc, vc, bt, ctx = build_paged(k, v, 16)

    peak = cudalib.peak_bandwidth(fresh=True)
    gbs, _ = cudalib.achieved_bandwidth(
        lambda: paged_attention_cuda(q, kc, vc, bt, ctx),
        kv_bytes(ctx, KVH, D, q.element_size()))

    print(f"\n  peak (measured now)  {peak:>7.0f} GB/s")
    print(f"  your kernel          {gbs:>7.0f} GB/s   "
          f"\033[1m{100 * gbs / peak:.0f}% of it\033[0m")
    print("  \033[2mOne thread per position walks head_dim on its own, so the 32")
    print("  threads of a warp ask for 32 scattered rows at once. The bytes")
    print("  arrive; the requests are what you overspent. Stage 08b.\033[0m")
