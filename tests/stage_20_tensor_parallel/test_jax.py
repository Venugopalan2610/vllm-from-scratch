"""Stage 20 (JAX) - Tensor parallelism with shard_map.

Spec in app/j20_tensor_parallel.py.

The mesh here is CPU devices, because one GPU is one device. Everything else --
the shardings, the psum, the partial sums -- is exactly what runs on eight
accelerators.
"""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from app.j20_tensor_parallel import (allreduce_bytes_per_token, make_mesh,
                                     shard_column, shard_heads, shard_row,
                                     tp_attention_forward, tp_mlp_forward)

HID, INTER, HEADS, HEAD_DIM = 32, 64, 8, 8


@pytest.fixture(scope="module")
def cpu():
    if len(jax.devices("cpu")) < 4:
        pytest.skip("need >= 4 CPU devices "
                    "(XLA_FLAGS=--xla_force_host_platform_device_count=8)")
    return jax.devices("cpu")


def _rand(shape, seed):
    return jnp.asarray(np.random.RandomState(seed).randn(*shape) * 0.1,
                       jnp.float32)


def test_shard_column_and_row(cpu):
    W = jnp.arange(8 * 4, dtype=jnp.float32).reshape(8, 4)
    assert shard_column(W, 0, 2).shape == (4, 4)
    assert shard_row(W, 0, 2).shape == (8, 2)
    np.testing.assert_array_equal(np.asarray(shard_column(W, 1, 2)),
                                  np.asarray(W)[4:])
    np.testing.assert_array_equal(np.asarray(shard_row(W, 1, 2)),
                                  np.asarray(W)[:, 2:])
    # the two together must reconstruct the original
    joined = jnp.concatenate([shard_column(W, r, 4) for r in range(4)], axis=0)
    np.testing.assert_array_equal(np.asarray(joined), np.asarray(W))


def test_shard_heads_splits_on_head_boundaries(cpu):
    """Whole heads, not raw rows. Splitting mid-head is silently wrong."""
    nh, hd, ws = 4, 8, 2
    W = jnp.arange(nh * hd * 3, dtype=jnp.float32).reshape(nh * hd, 3)
    got = shard_heads(W, 1, ws, nh, hd)
    assert got.shape == (2 * hd, 3)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(W)[2 * hd:])
    with pytest.raises(AssertionError):
        shard_heads(W, 0, 3, nh, hd)     # 4 heads do not divide by 3


@pytest.mark.parametrize("ws", [1, 2, 4])
def test_tp_mlp_matches_the_unsharded_answer(cpu, ws):
    mesh = make_mesh(ws)
    x = _rand((6, HID), 0)
    W1 = _rand((INTER, HID), 1)
    W2 = _rand((HID, INTER), 2)

    got = tp_mlp_forward(x, W1, W2, mesh)
    with jax.default_device(cpu[0]):
        want = jax.nn.gelu(x @ W1.T) @ W2.T
    np.testing.assert_allclose(np.asarray(got), np.asarray(want),
                               rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("ws", [1, 2, 4])
def test_tp_attention_matches_the_unsharded_answer(cpu, ws):
    mesh = make_mesh(ws)
    B, T = 2, 5
    x = _rand((B, T, HID), 3)
    Wq = _rand((HEADS * HEAD_DIM, HID), 4)
    Wk = _rand((HEADS * HEAD_DIM, HID), 5)
    Wv = _rand((HEADS * HEAD_DIM, HID), 6)
    Wo = _rand((HID, HEADS * HEAD_DIM), 7)

    got = tp_attention_forward(x, Wq, Wk, Wv, Wo, mesh, HEADS, HEAD_DIM)

    with jax.default_device(cpu[0]):
        def heads(w):
            return (x @ w.T).reshape(B, T, HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        q, k, v = heads(Wq), heads(Wk), heads(Wv)
        scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(HEAD_DIM)
        causal = jnp.tril(jnp.ones((T, T), bool))
        scores = jnp.where(causal, scores, jnp.finfo(scores.dtype).min)
        o = jax.nn.softmax(scores, -1) @ v
        want = o.transpose(0, 2, 1, 3).reshape(B, T, HEADS * HEAD_DIM) @ Wo.T

    np.testing.assert_allclose(np.asarray(got), np.asarray(want),
                               rtol=2e-4, atol=2e-5)


def test_exactly_one_all_reduce_per_block(cpu):
    """The property that makes tensor parallelism affordable.

    Two collectives per block means you all-reduced something that was already
    replicated. On real hardware that is the difference between TP being cheap
    and TP being pointless, and it is invisible in the numbers -- the answer is
    still correct, it just costs twice the interconnect.
    """
    mesh = make_mesh(2)
    x, W1, W2 = _rand((6, HID), 0), _rand((INTER, HID), 1), _rand((HID, INTER), 2)
    hlo = jax.jit(lambda a, b, c: tp_mlp_forward(a, b, c, mesh)) \
        .lower(x, W1, W2).as_text()
    # StableHLO spells it all_reduce; the compiled HLO spells it all-reduce.
    n = len(re.findall(r"\ball[_-]reduce\b", hlo))
    print(f"\n  collectives in the sharded MLP: {n}")
    assert n == 1, (
        f"{n} all-reduces in one MLP block. Column-parallel then row-parallel "
        "needs exactly one, at the end. Is x being reduced too, or are you "
        "psum-ing between the two matmuls?"
    )


def test_allreduce_bytes_per_token():
    assert allreduce_bytes_per_token(4096) == 4096 * 2 * 1 * 2
    assert allreduce_bytes_per_token(4096, 2, 32, 2) == 4096 * 2 * 32 * 2
    per_token = allreduce_bytes_per_token(4096, 2, 32, 2)
    print(f"\n  a 4096-hidden, 32-layer model all-reduces "
          f"{per_token / 1024:.0f} KB per token per rank")
    print("  \033[2mAt 100 tokens/s per sequence and batch 64 that is a few")
    print("  hundred MB/s -- fine over NVLink, and the reason tensor")
    print("  parallelism stops at the node boundary.\033[0m")


def test_uneven_shards_are_refused(cpu):
    """A mesh that does not divide the heads must fail loudly."""
    W = _rand((HEADS * HEAD_DIM, HID), 8)
    with pytest.raises(AssertionError):
        shard_heads(W, 0, 3, HEADS, HEAD_DIM)
    with pytest.raises(AssertionError):
        shard_column(_rand((7, HID), 9), 0, 2)
