"""Stage 07 (JAX) - Paged attention, the correct-and-slow reference.

Spec in app/j07_paged_attn.py.

The physical blocks are shuffled in every fixture here. An implementation that
assumes sequence-block b lives at physical block b passes nothing.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from app.j07_paged_attn import paged_attention, reference_attention, write_kv
from tests.jhelpers import build_paged, rand_kv


def close(got, want, tol=2e-3):
    np.testing.assert_allclose(np.asarray(got, np.float32),
                               np.asarray(want, np.float32),
                               rtol=tol, atol=tol)


def test_write_kv_scatters_to_the_right_slots(jdev):
    """A slot is block*block_size + offset, and blocks are not in order."""
    NB, KVH, BS, D = 6, 2, 4, 8
    kc = jnp.zeros((NB, KVH, BS, D))
    vc = jnp.zeros((NB, KVH, BS, D))
    slots = jnp.asarray([0, 5, 9, 23], jnp.int32)   # blocks 0, 1, 2, 5
    key = jnp.arange(4 * KVH * D, dtype=jnp.float32).reshape(4, KVH, D)
    val = key * -1

    kc, vc = write_kv(kc, vc, key, val, slots)
    for i, s in enumerate([0, 5, 9, 23]):
        b, o = s // BS, s % BS
        close(kc[b, :, o, :], key[i])
        close(vc[b, :, o, :], val[i])
    assert float(jnp.abs(kc).sum()) > 0, "wrote nothing"
    # everything else untouched
    assert float(jnp.abs(kc[3]).sum()) == 0.0, "wrote into a block you were not given"


@pytest.mark.parametrize("block_size", [1, 4, 16])
@pytest.mark.parametrize("L", [1, 7, 16, 33, 129])
def test_matches_the_dense_reference(jdev, block_size, L):
    """Every layout, every length, against contiguous K/V."""
    S, H, KVH, D = 3, 4, 4, 16
    k, v = rand_kv(S, KVH, L, D)
    q = jnp.asarray(np.random.RandomState(0).randn(S, H, D), jnp.float32)
    kc, vc, bt, ctx = build_paged(k, v, block_size)

    close(paged_attention(q, kc, vc, bt, ctx), reference_attention(q, k, v))


@pytest.mark.parametrize("H,KVH", [(8, 2), (16, 8), (4, 4)])
def test_gqa_ratios(jdev, H, KVH):
    """Query head h must read KV head h // (H // KVH)."""
    S, D, L = 2, 64, 40
    k, v = rand_kv(S, KVH, L, D)
    q = jnp.asarray(np.random.RandomState(1).randn(S, H, D), jnp.float32)
    kc, vc, bt, ctx = build_paged(k, v, 16)
    close(paged_attention(q, kc, vc, bt, ctx), reference_attention(q, k, v))


def test_ragged_context_lengths(jdev):
    """Different lengths in one call, all reading the same block table."""
    S, H, KVH, D, Lmax = 8, 8, 4, 64, 200
    k, v = rand_kv(S, KVH, Lmax, D)
    kc, vc, bt, _ = build_paged(k, v, 16)
    lens = jnp.asarray([200, 1, 17, 16, 199, 33, 64, 128], jnp.int32)
    q = jnp.asarray(np.random.RandomState(2).randn(S, H, D), jnp.float32)

    got = paged_attention(q, kc, vc, bt, lens)
    for i in range(S):
        n = int(lens[i])
        want = reference_attention(q[i: i + 1], k[i: i + 1, :, :n],
                                   v[i: i + 1, :, :n])
        close(got[i: i + 1], want)


def test_ignores_junk_beyond_context_len(jdev):
    """Blocks are recycled, so the tail of the last block is someone else's data.

    Mask on `position < context_len`, not on the block count.
    """
    S, H, KVH, D, L = 2, 4, 2, 32, 12
    bs = 8
    k, v = rand_kv(S, KVH, L, D)
    kc, vc, bt, _ = build_paged(k, v, bs)
    lens = jnp.asarray([L, L], jnp.int32)
    q = jnp.asarray(np.random.RandomState(3).randn(S, H, D), jnp.float32)
    before = paged_attention(q, kc, vc, bt, lens)

    kc_np, vc_np = np.asarray(kc).copy(), np.asarray(vc).copy()
    for s in range(S):
        for b in range(bt.shape[1]):
            phys = int(bt[s, b])
            for off in range(bs):
                if b * bs + off >= L:
                    kc_np[phys, :, off] = 999.0
                    vc_np[phys, :, off] = 999.0

    after = paged_attention(q, jnp.asarray(kc_np), jnp.asarray(vc_np), bt, lens)
    close(after, before)


def test_bf16_cache_accumulates_in_fp32(jdev):
    """A long bf16 context is where sloppy accumulation shows up."""
    S, H, KVH, D, L = 2, 8, 8, 128, 1024
    k, v = rand_kv(S, KVH, L, D, dtype=jnp.bfloat16)
    q = jnp.asarray(np.random.RandomState(4).randn(S, H, D), jnp.bfloat16)
    kc, vc, bt, ctx = build_paged(k, v, 16)

    got = paged_attention(q, kc, vc, bt, ctx)
    want = reference_attention(q.astype(jnp.float32), k.astype(jnp.float32),
                               v.astype(jnp.float32))
    close(got, want, tol=6e-2)


def test_the_slowdown_you_just_ate(jdev):
    """Not pass/fail. Paging costs you something; see what."""
    from tests.jhelpers import jbench_ms

    S, H, KVH, D, L = 32, 16, 8, 128, 512
    k, v = rand_kv(S, KVH, L, D, dtype=jnp.bfloat16)
    q = jnp.asarray(np.random.RandomState(5).randn(S, H, D), jnp.bfloat16)
    kc, vc, bt, ctx = build_paged(k, v, 16)

    dense = jbench_ms(lambda: reference_attention(q, k, v))
    paged = jbench_ms(lambda: paged_attention(q, kc, vc, bt, ctx))
    print(f"\n  dense (contiguous K/V): {dense:7.3f} ms")
    print(f"  paged (gather first):  {paged:7.3f} ms   {paged / dense:.2f}x")
    print("\n  \033[2mThe gather is a full copy of every sequence's K and V,")
    print("  written to HBM and read straight back. Stage 08 deletes it by")
    print("  never materialising the gathered cache at all.\033[0m")
