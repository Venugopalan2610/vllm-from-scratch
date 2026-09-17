"""Stage 08 (JAX) - Paged attention in Pallas.

Spec in app/j08_paged_pallas.py.

Every correctness check compares against stage 07, which is your oracle. The
last one is the one that matters: it has to actually be faster.

These checks need a working Pallas GPU backend. On a pre-Hopper card that is
the Triton backend, which `jvllm.compat` registers for you; if neither backend
can compile here, the whole file skips rather than failing.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from app.j07_paged_attn import paged_attention, reference_attention
from app.j08_paged_pallas import paged_attention_pallas
from tests.jhelpers import build_paged, jbench_ms, rand_kv


def close(got, want, tol=2e-3):
    np.testing.assert_allclose(np.asarray(got, np.float32),
                               np.asarray(want, np.float32),
                               rtol=tol, atol=tol)


@pytest.mark.parametrize("block_size", [1, 4, 16])
@pytest.mark.parametrize("L", [1, 7, 16, 33, 129])
def test_agrees_with_stage_07(jpallas, block_size, L):
    """Same numbers as the jnp version, every layout, every length."""
    S, H, KVH, D = 3, 4, 4, 16
    k, v = rand_kv(S, KVH, L, D)
    q = jnp.asarray(np.random.RandomState(0).randn(S, H, D), jnp.float32)
    kc, vc, bt, ctx = build_paged(k, v, block_size)
    close(paged_attention_pallas(q, kc, vc, bt, ctx),
          paged_attention(q, kc, vc, bt, ctx))


def test_agrees_with_the_dense_reference_too(jpallas):
    """Belt and braces: not just consistent with stage 07, but correct."""
    S, H, KVH, D, L = 4, 8, 8, 64, 100
    k, v = rand_kv(S, KVH, L, D)
    q = jnp.asarray(np.random.RandomState(1).randn(S, H, D), jnp.float32)
    kc, vc, bt, ctx = build_paged(k, v, 16)
    close(paged_attention_pallas(q, kc, vc, bt, ctx),
          reference_attention(q, k, v))


@pytest.mark.parametrize("H,KVH", [(8, 2), (16, 8), (4, 4)])
def test_gqa_ratios(jpallas, H, KVH):
    """Query head h must read KV head h // (H // KVH)."""
    S, D, L = 2, 64, 40
    k, v = rand_kv(S, KVH, L, D)
    q = jnp.asarray(np.random.RandomState(2).randn(S, H, D), jnp.float32)
    kc, vc, bt, ctx = build_paged(k, v, 16)
    close(paged_attention_pallas(q, kc, vc, bt, ctx),
          paged_attention(q, kc, vc, bt, ctx))


def test_ragged_context_lengths(jpallas):
    """The fori_loop trip count is per program, and it is a device value."""
    S, H, KVH, D, Lmax = 8, 8, 4, 64, 200
    k, v = rand_kv(S, KVH, Lmax, D)
    kc, vc, bt, _ = build_paged(k, v, 16)
    lens = jnp.asarray([200, 1, 17, 16, 199, 33, 64, 128], jnp.int32)
    q = jnp.asarray(np.random.RandomState(3).randn(S, H, D), jnp.float32)
    close(paged_attention_pallas(q, kc, vc, bt, lens),
          paged_attention(q, kc, vc, bt, lens))


def test_ignores_junk_beyond_context_len(jpallas):
    """The masking check again, because a kernel is much easier to get wrong."""
    S, H, KVH, D, L = 2, 4, 2, 32, 12
    bs = 8
    k, v = rand_kv(S, KVH, L, D)
    kc, vc, bt, _ = build_paged(k, v, bs)
    lens = jnp.asarray([L, L], jnp.int32)
    q = jnp.asarray(np.random.RandomState(4).randn(S, H, D), jnp.float32)
    before = paged_attention_pallas(q, kc, vc, bt, lens)

    kc_np, vc_np = np.asarray(kc).copy(), np.asarray(vc).copy()
    for s in range(S):
        for b in range(bt.shape[1]):
            phys = int(bt[s, b])
            for off in range(bs):
                if b * bs + off >= L:
                    kc_np[phys, :, off] = 999.0
                    vc_np[phys, :, off] = 999.0

    after = paged_attention_pallas(q, jnp.asarray(kc_np), jnp.asarray(vc_np),
                                   bt, lens)
    close(after, before)


def test_bf16_accumulates_in_fp32(jpallas):
    """Long contexts in bf16 are where sloppy accumulation shows up.

    Sum 2048 bf16 products in bf16 and you lose several digits. Accumulate in
    float32 inside the kernel and this passes comfortably.
    """
    S, H, KVH, D, L = 2, 8, 8, 128, 2048
    k, v = rand_kv(S, KVH, L, D, dtype=jnp.bfloat16)
    q = jnp.asarray(np.random.RandomState(5).randn(S, H, D), jnp.bfloat16)
    kc, vc, bt, ctx = build_paged(k, v, 16)
    want = reference_attention(q.astype(jnp.float32), k.astype(jnp.float32),
                               v.astype(jnp.float32))
    close(paged_attention_pallas(q, kc, vc, bt, ctx), want, tol=6e-2)


def test_it_is_actually_faster(jpallas):
    """The whole point of the stage."""
    rows = []
    for S, L in ((8, 256), (32, 512), (64, 1024)):
        H, KVH, D = 16, 8, 128
        k, v = rand_kv(S, KVH, L, D, dtype=jnp.bfloat16)
        q = jnp.asarray(np.random.RandomState(6).randn(S, H, D), jnp.bfloat16)
        kc, vc, bt, ctx = build_paged(k, v, 16)

        jnp_ms = jbench_ms(lambda: paged_attention(q, kc, vc, bt, ctx), iters=10)
        pallas_ms = jbench_ms(lambda: paged_attention_pallas(q, kc, vc, bt, ctx))
        rows.append((S, L, jnp_ms, pallas_ms))

    print(f"\n  {'seqs':>5} {'ctx':>6} {'stage 07':>11} {'stage 08':>11} {'speedup':>9}")
    for S, L, a, b in rows:
        print(f"  {S:>5} {L:>6} {a:>9.2f}ms {b:>9.3f}ms {a / b:>8.1f}x")

    worst = min(a / b for _, _, a, b in rows)
    assert worst > 3.0, (
        f"only {worst:.1f}x faster at best. Stage 07 copies every sequence's "
        "whole K and V to HBM before it computes anything; one kernel that "
        "streams tiles should crush it."
    )
    print(f"\n  \033[1mWorst case: {worst:.0f}x faster than the jnp version.\033[0m")
    print("  \033[2mYou deleted the gather. The K/V tiles now stream through")
    print("  SRAM instead of round-tripping to HBM as a materialised copy,")
    print("  and the score row never exists at all.\033[0m")
