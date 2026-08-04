"""Stage 07 - Attention that reads through the page table.

Spec in app/s07_paged_attn.py.

The contract that matters: paged_attention() must agree with a dense reference
to floating-point tolerance, for every block layout, every context length, and
GQA configurations. If it doesn't, stage 08's Triton kernel has no oracle and
you will be debugging two things at once.
"""

import math
import random

import pytest
import torch

from app.s07_paged_attn import paged_attention, reference_attention, write_kv


def build_paged(keys, values, block_size, shuffle=True, seed=0):
    """Scatter dense (S, KVH, L, D) K/V into a paged cache with a random
    block layout. Returns (key_cache, value_cache, block_tables, context_lens).

    Deliberately shuffles physical blocks -- a correct implementation must not
    care what order they land in.
    """
    S, KVH, L, D = keys.shape
    nb_per_seq = math.ceil(L / block_size)
    total = S * nb_per_seq
    ids = list(range(total))
    if shuffle:
        random.Random(seed).shuffle(ids)

    kc = torch.zeros(total, KVH, block_size, D, dtype=keys.dtype, device=keys.device)
    vc = torch.zeros_like(kc)
    bt = torch.zeros(S, nb_per_seq, dtype=torch.int32, device=keys.device)

    for s in range(S):
        for b in range(nb_per_seq):
            phys = ids[s * nb_per_seq + b]
            bt[s, b] = phys
            lo, hi = b * block_size, min((b + 1) * block_size, L)
            kc[phys, :, : hi - lo] = keys[s, :, lo:hi]
            vc[phys, :, : hi - lo] = values[s, :, lo:hi]

    ctx = torch.full((S,), L, dtype=torch.int32, device=keys.device)
    return kc, vc, bt, ctx


def rand_kv(S, KVH, L, D, dev, dtype=torch.float32):
    g = torch.Generator(device=dev).manual_seed(1234)
    k = torch.randn(S, KVH, L, D, generator=g, device=dev, dtype=dtype)
    v = torch.randn(S, KVH, L, D, generator=g, device=dev, dtype=dtype)
    return k, v


# ---- write_kv -------------------------------------------------------

def test_write_kv_scatters_to_the_right_slots(dev):
    block_size, D, KVH = 4, 8, 2
    kc = torch.zeros(6, KVH, block_size, D, device=dev)
    vc = torch.zeros_like(kc)
    k = torch.randn(3, KVH, D, device=dev)
    v = torch.randn(3, KVH, D, device=dev)
    # slots 5, 0, 21 -> (block 1, off 1), (block 0, off 0), (block 5, off 1)
    slots = torch.tensor([5, 0, 21], device=dev)
    write_kv(kc, vc, k, v, slots)

    assert torch.allclose(kc[1, :, 1], k[0])
    assert torch.allclose(kc[0, :, 0], k[1])
    assert torch.allclose(kc[5, :, 1], k[2])
    assert torch.allclose(vc[5, :, 1], v[2])
    assert kc[2].abs().sum() == 0, "wrote to a block that was not targeted"


# ---- correctness ----------------------------------------------------

@pytest.mark.parametrize("block_size", [1, 4, 16])
@pytest.mark.parametrize("L", [1, 7, 16, 33])
def test_matches_dense_reference(dev, block_size, L):
    """Every block size, including L not a multiple of block_size."""
    S, H, KVH, D = 3, 4, 4, 16
    k, v = rand_kv(S, KVH, L, D, dev)
    q = torch.randn(S, H, D, device=dev)

    want = reference_attention(q, k, v)
    kc, vc, bt, ctx = build_paged(k, v, block_size)
    got = paged_attention(q, kc, vc, bt, ctx)

    assert got.shape == want.shape
    torch.testing.assert_close(got, want, rtol=2e-3, atol=2e-3)


def test_gqa(dev):
    """num_heads != num_kv_heads. Query head h reads KV head h // group.

    Get the ratio backwards and the output still looks like attention output.
    It is just wrong. This is the single most common bug in this stage.
    """
    S, H, KVH, D, L = 2, 8, 2, 16, 20   # group = 4
    k, v = rand_kv(S, KVH, L, D, dev)
    q = torch.randn(S, H, D, device=dev)

    want = reference_attention(q, k, v)
    kc, vc, bt, ctx = build_paged(k, v, block_size=8)
    got = paged_attention(q, kc, vc, bt, ctx)
    torch.testing.assert_close(got, want, rtol=2e-3, atol=2e-3)


def test_ragged_context_lengths(dev):
    """Sequences in one batch have different lengths. That is the normal case."""
    S, H, KVH, D, Lmax = 4, 4, 2, 16, 40
    block_size = 8
    k, v = rand_kv(S, KVH, Lmax, D, dev)
    kc, vc, bt, _ = build_paged(k, v, block_size)
    lens = torch.tensor([40, 1, 17, 32], dtype=torch.int32, device=dev)
    q = torch.randn(S, H, D, device=dev)

    got = paged_attention(q, kc, vc, bt, lens)
    for i, L in enumerate(lens.tolist()):
        want_i = reference_attention(q[i:i + 1], k[i:i + 1, :, :L], v[i:i + 1, :, :L])
        torch.testing.assert_close(got[i:i + 1], want_i, rtol=2e-3, atol=2e-3)


def test_ignores_junk_beyond_context_len(dev):
    """Freed blocks get reused and hold another sequence's garbage.

    If you attend past context_lens you will silently mix in a different
    request's data -- a correctness bug AND a data-leak between users.
    """
    S, H, KVH, D, L = 2, 4, 2, 16, 12
    block_size = 8
    k, v = rand_kv(S, KVH, L, D, dev)
    kc, vc, bt, _ = build_paged(k, v, block_size)
    lens = torch.tensor([L, L], dtype=torch.int32, device=dev)
    q = torch.randn(S, H, D, device=dev)
    before = paged_attention(q, kc, vc, bt, lens)

    # poison every slot at or beyond context_len
    for s in range(S):
        for b in range(bt.shape[1]):
            phys = int(bt[s, b])
            for off in range(block_size):
                if b * block_size + off >= L:
                    kc[phys, :, off] = 999.0
                    vc[phys, :, off] = 999.0

    after = paged_attention(q, kc, vc, bt, lens)
    torch.testing.assert_close(before, after, rtol=2e-3, atol=2e-3)


def test_physical_block_order_is_irrelevant(dev):
    """Same logical content, different physical layout -> identical output."""
    S, H, KVH, D, L = 2, 4, 4, 16, 24
    k, v = rand_kv(S, KVH, L, D, dev)
    q = torch.randn(S, H, D, device=dev)
    a = paged_attention(q, *build_paged(k, v, 8, shuffle=True, seed=1))
    b = paged_attention(q, *build_paged(k, v, 8, shuffle=True, seed=2))
    torch.testing.assert_close(a, b, rtol=2e-3, atol=2e-3)


def test_softmax_rows_sum_to_one(dev):
    """A cheap invariant that catches masking bugs the reference might share."""
    S, H, KVH, D, L = 2, 4, 4, 8, 16
    k, v = rand_kv(S, KVH, L, D, dev)
    kc, vc, bt, ctx = build_paged(k, v, 4)
    # with V = all ones, attention output must be exactly 1 everywhere
    vc = torch.ones_like(vc)
    q = torch.randn(S, H, D, device=dev)
    out = paged_attention(q, kc, vc, bt, ctx)
    torch.testing.assert_close(out, torch.ones_like(out), rtol=1e-3, atol=1e-3)


def test_report_the_slowdown(dev):
    """Not pass/fail. Paging cost you something -- find out how much."""
    import time
    S, H, KVH, D, L = 64, 16, 8, 128, 512
    k, v = rand_kv(S, KVH, L, D, dev, dtype=torch.float16)
    q = torch.randn(S, H, D, device=dev, dtype=torch.float16)
    kc, vc, bt, ctx = build_paged(k, v, 16)

    def bench(fn, n=10):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / n * 1000

    dense = bench(lambda: reference_attention(q, k, v))
    paged = bench(lambda: paged_attention(q, kc, vc, bt, ctx))
    print(f"\n  dense reference: {dense:7.2f} ms")
    print(f"  paged (PyTorch): {paged:7.2f} ms   ({paged / dense:.1f}x slower)")
    print("\n  \033[2mThe python loop over sequences is the problem. Stage 08")
    print("  replaces it with one Triton kernel and wins this back.\033[0m")
