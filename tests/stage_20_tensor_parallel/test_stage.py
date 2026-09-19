"""Stage 20 - tensor parallelism (simulated on one GPU).

The spec is in app/s20_tensor_parallel.py. Correctness comes first. One GPU
gives no speedup, and that is acceptable. The lesson is where the
communication occurs.
"""

import pytest
import torch
import torch.nn.functional as F

from app.s20_tensor_parallel import (
    allreduce_bytes_per_token,
    run_distributed_mlp,
    shard_column,
    shard_heads,
    shard_row,
    tp_attention_forward,
    tp_mlp_forward,
)


def mlp_problem():
    """-> (inputs, up_weight, down_weight, the expected MLP output)."""
    torch.manual_seed(0)
    inputs = torch.randn(4, 64)
    up_weight = torch.randn(256, 64) / 8
    down_weight = torch.randn(64, 256) / 16
    return (inputs, up_weight, down_weight,
            F.gelu(inputs @ up_weight.T) @ down_weight.T)


def test_shard_shapes():
    weight = torch.arange(32).reshape(8, 4).float()
    assert shard_column(weight, 0, 2).shape == (4, 4)
    assert shard_row(weight, 0, 2).shape == (8, 2)
    # The shards must cover the original exactly: no overlap, no gaps.
    torch.testing.assert_close(
        torch.cat([shard_column(weight, rank, 2) for rank in range(2)], dim=0),
        weight)
    torch.testing.assert_close(
        torch.cat([shard_row(weight, rank, 2) for rank in range(2)], dim=1),
        weight)


def test_shard_heads_splits_on_head_boundaries():
    num_heads, head_dim, hidden = 8, 16, 128
    weight = torch.randn(num_heads * head_dim, hidden)
    first = shard_heads(weight, 0, 2, num_heads, head_dim)
    second = shard_heads(weight, 1, 2, num_heads, head_dim)
    assert first.shape == (4 * head_dim, hidden)
    torch.testing.assert_close(first, weight[:4 * head_dim])
    torch.testing.assert_close(second, weight[4 * head_dim:])


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
def test_sharded_mlp_matches_unsharded(world_size):
    """The only important correctness condition: the same answer at every
    world size."""
    inputs, up_weight, down_weight, expected = mlp_problem()
    torch.testing.assert_close(
        tp_mlp_forward(inputs, up_weight, down_weight, world_size), expected,
        rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("world_size", [2, 4])
def test_sharded_attention_matches_unsharded(world_size):
    torch.manual_seed(0)
    num_heads, head_dim = 8, 16
    hidden = num_heads * head_dim
    inputs = torch.randn(2, 10, hidden)
    projections = [torch.randn(hidden, hidden) / 8 for _ in range(4)]
    expected = tp_attention_forward(inputs, *projections, 1, num_heads,
                                    head_dim)
    torch.testing.assert_close(
        tp_attention_forward(inputs, *projections, world_size, num_heads,
                             head_dim),
        expected, rtol=1e-4, atol=1e-4)


def test_splitting_mid_head_is_wrong():
    """Shows the usual bug, so that you know it.

    A cut of the Q, K or V projection at a random row, not at a head
    boundary, mixes parts of different heads. The output has the correct
    shape and a correct size, and it is wrong.
    """
    torch.manual_seed(0)
    num_heads, head_dim = 4, 8
    hidden = num_heads * head_dim
    weight = torch.randn(hidden, hidden)
    correct = shard_heads(weight, 0, 2, num_heads, head_dim)
    three_rows_off = weight[3:3 + correct.shape[0]]
    assert three_rows_off.shape == correct.shape, "the same shape..."
    assert not torch.allclose(three_rows_off, correct), "...different weights"


def test_communication_accounting():
    # hidden 4096, bf16, 32 layers, 2 all-reduces in each layer
    bytes_per_token = allreduce_bytes_per_token(4096, 2, 32, 2)
    assert bytes_per_token == 4096 * 2 * 32 * 2
    print("\n  hidden 4096, bf16, 32 layers, 2 all-reduces in each layer:")
    print(f"    {bytes_per_token / 1024:.0f} KB for each token on each rank")
    print(f"    at 4000 tok/s that is {bytes_per_token * 4000 / 1e9:.1f} GB/s "
          "of collective traffic")
    print("\n  \033[2mNVLink carries that easily. PCIe or Ethernet does not.")
    print("  That is why tensor parallelism is used INSIDE a node, and")
    print("  pipeline parallelism between nodes.\033[0m")


@pytest.mark.timeout(180)
def test_real_two_rank_all_reduce():
    """Real processes, a real dist.all_reduce. gloo on the CPU, because you
    have 1 GPU.

    On hardware with many GPUs this is NCCL, and the call is the same.
    """
    inputs, up_weight, down_weight, expected = mlp_problem()
    torch.testing.assert_close(
        run_distributed_mlp(up_weight, down_weight, inputs, world_size=2),
        expected, rtol=1e-4, atol=1e-5)
    print("\n  2 started ranks, a gloo all-reduce, and the output is the same "
          "as one rank")
    print("  \033[2mYou built every major part of a modern inference engine.")
    print("  Read the vLLM source now. It is familiar.\033[0m")
