"""Stage 08 - Paged attention in Triton.

Spec in app/s08_paged_triton.py.

Every correctness test here compares against stage 07, which is your oracle.
The last test is the one that matters: it has to actually be faster.
"""

import pytest
import torch

from app.s07_paged_attn import paged_attention, reference_attention
from app.s08_paged_triton import paged_attention_triton
from tests.helpers import bench_ms, build_paged, rand_kv


@pytest.mark.parametrize("block_size", [1, 4, 16])
@pytest.mark.parametrize("L", [1, 7, 16, 33, 129])
def test_agrees_with_stage_07(dev, block_size, L):
    """Same numbers as the PyTorch version, every layout, every length."""
    S, H, KVH, D = 3, 4, 4, 16
    k, v = rand_kv(S, KVH, L, D, dev)
    q = torch.randn(S, H, D, device=dev)
    kc, vc, bt, ctx = build_paged(k, v, block_size)

    want = paged_attention(q, kc, vc, bt, ctx)
    got = paged_attention_triton(q, kc, vc, bt, ctx)
    torch.testing.assert_close(got, want, rtol=1e-3, atol=1e-3)


def test_agrees_with_the_dense_reference_too(dev):
    """Belt and braces: not just consistent with stage 07, but correct."""
    S, H, KVH, D, L = 4, 8, 8, 64, 100
    k, v = rand_kv(S, KVH, L, D, dev)
    q = torch.randn(S, H, D, device=dev)
    kc, vc, bt, ctx = build_paged(k, v, 16)
    want = reference_attention(q, k, v)
    got = paged_attention_triton(q, kc, vc, bt, ctx)
    torch.testing.assert_close(got, want, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("H,KVH", [(8, 2), (16, 8), (4, 4)])
def test_gqa_ratios(dev, H, KVH):
    """Query head h must read KV head h // (H // KVH)."""
    S, D, L = 2, 64, 40
    k, v = rand_kv(S, KVH, L, D, dev)
    q = torch.randn(S, H, D, device=dev)
    kc, vc, bt, ctx = build_paged(k, v, 16)
    want = paged_attention(q, kc, vc, bt, ctx)
    got = paged_attention_triton(q, kc, vc, bt, ctx)
    torch.testing.assert_close(got, want, rtol=1e-3, atol=1e-3)


def test_ragged_context_lengths(dev):
    S, H, KVH, D, Lmax = 8, 8, 4, 64, 200
    k, v = rand_kv(S, KVH, Lmax, D, dev)
    kc, vc, bt, _ = build_paged(k, v, 16)
    lens = torch.tensor([200, 1, 17, 16, 199, 33, 64, 128],
                        dtype=torch.int32, device=dev)
    q = torch.randn(S, H, D, device=dev)
    want = paged_attention(q, kc, vc, bt, lens)
    got = paged_attention_triton(q, kc, vc, bt, lens)
    torch.testing.assert_close(got, want, rtol=1e-3, atol=1e-3)


def test_ignores_junk_beyond_context_len(dev):
    """The masking test again, because a kernel is much easier to get wrong."""
    S, H, KVH, D, L = 2, 4, 2, 32, 12
    bs = 8
    k, v = rand_kv(S, KVH, L, D, dev)
    kc, vc, bt, _ = build_paged(k, v, bs)
    lens = torch.tensor([L, L], dtype=torch.int32, device=dev)
    q = torch.randn(S, H, D, device=dev)
    before = paged_attention_triton(q, kc, vc, bt, lens)

    for s in range(S):
        for b in range(bt.shape[1]):
            phys = int(bt[s, b])
            for off in range(bs):
                if b * bs + off >= L:
                    kc[phys, :, off] = 999.0
                    vc[phys, :, off] = 999.0

    after = paged_attention_triton(q, kc, vc, bt, lens)
    torch.testing.assert_close(before, after, rtol=1e-3, atol=1e-3)


def test_fp16_accumulates_in_fp32(dev):
    """Long contexts in fp16 are where sloppy accumulation shows up.

    Sum 2048 fp16 products in fp16 and you lose several digits. Accumulate in
    float32 inside the kernel and this passes comfortably.
    """
    S, H, KVH, D, L = 2, 8, 8, 128, 2048
    k, v = rand_kv(S, KVH, L, D, dev, dtype=torch.float16)
    q = torch.randn(S, H, D, device=dev, dtype=torch.float16)
    kc, vc, bt, ctx = build_paged(k, v, 16)
    want = reference_attention(q, k, v)
    got = paged_attention_triton(q, kc, vc, bt, ctx)
    torch.testing.assert_close(got, want, rtol=3e-3, atol=3e-3)


def test_it_is_actually_faster(dev):
    """The whole point of the stage."""
    rows = []
    for S, L in ((8, 256), (32, 512), (64, 1024)):
        H, KVH, D = 16, 8, 128
        k, v = rand_kv(S, KVH, L, D, dev, dtype=torch.float16)
        q = torch.randn(S, H, D, device=dev, dtype=torch.float16)
        kc, vc, bt, ctx = build_paged(k, v, 16)

        torch_ms = bench_ms(lambda: paged_attention(q, kc, vc, bt, ctx), iters=10)
        triton_ms = bench_ms(lambda: paged_attention_triton(q, kc, vc, bt, ctx))
        rows.append((S, L, torch_ms, triton_ms))

    print(f"\n  {'seqs':>5} {'ctx':>6} {'stage 07':>11} {'stage 08':>11} {'speedup':>9}")
    for S, L, a, b in rows:
        print(f"  {S:>5} {L:>6} {a:>9.2f}ms {b:>9.3f}ms {a / b:>8.0f}x")

    worst = min(a / b for _, _, a, b in rows)
    assert worst > 3.0, (
        f"only {worst:.1f}x faster at best. One kernel launch should crush a "
        "Python loop that does num_seqs separate gathers."
    )
    print(f"\n  \033[1mWorst case: {worst:.0f}x faster than the PyTorch version.\033[0m")
    print("  \033[2mYou deleted num_seqs kernel launches and num_seqs gathers,")
    print("  and the K/V tiles now stream through SRAM instead of round-tripping")
    print("  to HBM as materialised score matrices.\033[0m")
