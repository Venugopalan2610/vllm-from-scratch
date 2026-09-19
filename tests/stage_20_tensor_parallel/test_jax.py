"""Stage 20 (JAX) - tensor parallelism with shard_map.

The spec is in app/j20_tensor_parallel.py.

The mesh here holds CPU devices, because one GPU is one device. All the rest
is what runs on eight accelerators: the shardings, the psum and the partial
sums.
"""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from app.j20_tensor_parallel import (
    allreduce_bytes_per_token,
    make_mesh,
    shard_column,
    shard_heads,
    shard_row,
    tp_attention_forward,
    tp_mlp_forward,
)

HIDDEN, INTERMEDIATE, NUM_HEADS, HEAD_DIM = 32, 64, 8, 8


@pytest.fixture(scope="module")
def cpu_devices():
    if len(jax.devices("cpu")) < 4:
        pytest.skip("need at least 4 CPU devices "
                    "(XLA_FLAGS=--xla_force_host_platform_device_count=8)")
    return jax.devices("cpu")


def _random(shape, seed):
    return jnp.asarray(np.random.RandomState(seed).randn(*shape) * 0.1,
                       jnp.float32)


def _mlp_weights():
    """-> (inputs, up_weight, down_weight)."""
    return (_random((6, HIDDEN), 0), _random((INTERMEDIATE, HIDDEN), 1),
            _random((HIDDEN, INTERMEDIATE), 2))


def _dense_causal_attention(inputs, query_weight, key_weight, value_weight,
                            output_weight):
    batch, seq_len, _ = inputs.shape

    def heads(weight):
        return (inputs @ weight.T).reshape(batch, seq_len, NUM_HEADS,
                                           HEAD_DIM).transpose(0, 2, 1, 3)

    query, key, value = heads(query_weight), heads(key_weight), heads(value_weight)
    scores = query @ key.transpose(0, 1, 3, 2) / np.sqrt(HEAD_DIM)
    causal = jnp.tril(jnp.ones((seq_len, seq_len), bool))
    scores = jnp.where(causal, scores, jnp.finfo(scores.dtype).min)
    attended = jax.nn.softmax(scores, -1) @ value
    return attended.transpose(0, 2, 1, 3).reshape(
        batch, seq_len, NUM_HEADS * HEAD_DIM) @ output_weight.T


def test_shard_column_and_row(cpu_devices):
    weight = jnp.arange(8 * 4, dtype=jnp.float32).reshape(8, 4)
    assert shard_column(weight, 0, 2).shape == (4, 4)
    assert shard_row(weight, 0, 2).shape == (8, 2)
    np.testing.assert_array_equal(np.asarray(shard_column(weight, 1, 2)),
                                  np.asarray(weight)[4:])
    np.testing.assert_array_equal(np.asarray(shard_row(weight, 1, 2)),
                                  np.asarray(weight)[:, 2:])
    # The shards together must make the original again.
    joined = jnp.concatenate([shard_column(weight, rank, 4)
                              for rank in range(4)], axis=0)
    np.testing.assert_array_equal(np.asarray(joined), np.asarray(weight))


def test_shard_heads_splits_on_head_boundaries(cpu_devices):
    """Shard whole heads, not raw rows. A shard in the middle of a head is
    wrong, and it gives no error."""
    num_heads, head_dim = 4, 8
    weight = jnp.arange(num_heads * head_dim * 3,
                        dtype=jnp.float32).reshape(num_heads * head_dim, 3)
    second = shard_heads(weight, 1, 2, num_heads, head_dim)
    assert second.shape == (2 * head_dim, 3)
    np.testing.assert_array_equal(np.asarray(second),
                                  np.asarray(weight)[2 * head_dim:])
    with pytest.raises(AssertionError):
        shard_heads(weight, 0, 3, num_heads, head_dim)   # 4 heads, 3 ranks


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_tp_mlp_matches_the_unsharded_answer(cpu_devices, world_size):
    inputs, up_weight, down_weight = _mlp_weights()
    output = tp_mlp_forward(inputs, up_weight, down_weight,
                            make_mesh(world_size))
    with jax.default_device(cpu_devices[0]):
        expected = jax.nn.gelu(inputs @ up_weight.T) @ down_weight.T
    np.testing.assert_allclose(np.asarray(output), np.asarray(expected),
                               rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_tp_attention_matches_the_unsharded_answer(cpu_devices, world_size):
    inputs = _random((2, 5, HIDDEN), 3)
    projections = (_random((NUM_HEADS * HEAD_DIM, HIDDEN), 4),
                   _random((NUM_HEADS * HEAD_DIM, HIDDEN), 5),
                   _random((NUM_HEADS * HEAD_DIM, HIDDEN), 6),
                   _random((HIDDEN, NUM_HEADS * HEAD_DIM), 7))
    output = tp_attention_forward(inputs, *projections, make_mesh(world_size),
                                  NUM_HEADS, HEAD_DIM)
    with jax.default_device(cpu_devices[0]):
        expected = _dense_causal_attention(inputs, *projections)
    np.testing.assert_allclose(np.asarray(output), np.asarray(expected),
                               rtol=2e-4, atol=2e-5)


def test_exactly_one_all_reduce_per_block(cpu_devices):
    """The property that makes tensor parallelism affordable.

    Two collectives in a block means that you all-reduced something that
    was already on every rank. On real hardware that decides if TP is cheap
    or useless. The numbers do not show it: the answer is still correct, and
    it only costs twice the interconnect.
    """
    mesh = make_mesh(2)
    hlo = jax.jit(lambda inputs, up_weight, down_weight: tp_mlp_forward(
        inputs, up_weight, down_weight, mesh)).lower(*_mlp_weights()).as_text()
    # StableHLO writes all_reduce. The compiled HLO writes all-reduce.
    num_all_reduces = len(re.findall(r"\ball[_-]reduce\b", hlo))
    print(f"\n  collectives in the sharded MLP: {num_all_reduces}")
    assert num_all_reduces == 1, (
        f"{num_all_reduces} all-reduces in one MLP block. Column-parallel, then "
        "row-parallel, needs exactly one, at the end. Do you also reduce the "
        "inputs, or psum between the two matmuls?")


def test_allreduce_bytes_per_token():
    assert allreduce_bytes_per_token(4096) == 4096 * 2 * 1 * 2
    bytes_per_token = allreduce_bytes_per_token(4096, 2, 32, 2)
    assert bytes_per_token == 4096 * 2 * 32 * 2
    print(f"\n  a model with hidden 4096 and 32 layers all-reduces "
          f"{bytes_per_token / 1024:.0f} KB for each token on each rank")
    print("  \033[2mAt 100 tokens/s for each sequence and batch 64, that is a")
    print("  few hundred MB/s. NVLink carries it, and it is the reason that")
    print("  tensor parallelism stops at the node boundary.\033[0m")


def test_uneven_shards_are_refused(cpu_devices):
    """A mesh that does not divide the heads must fail with a clear error."""
    with pytest.raises(AssertionError):
        shard_heads(_random((NUM_HEADS * HEAD_DIM, HIDDEN), 8), 0, 3,
                    NUM_HEADS, HEAD_DIM)
    with pytest.raises(AssertionError):
        shard_column(_random((7, HIDDEN), 9), 0, 2)
