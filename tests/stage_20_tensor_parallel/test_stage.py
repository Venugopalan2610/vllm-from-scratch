"""Stage 20 - Tensor parallelism (simulated on one GPU).

Spec in app/s20_tensor_parallel.py. Correctness first; there is no speedup to
be had on a single GPU, and that is fine -- the lesson is where the
communication lands.
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


def test_shard_shapes():
    W = torch.arange(32).reshape(8, 4).float()
    assert shard_column(W, 0, 2).shape == (4, 4)
    assert shard_row(W, 0, 2).shape == (8, 2)
    # shards must tile the original exactly, no overlap and no gaps
    torch.testing.assert_close(
        torch.cat([shard_column(W, r, 2) for r in range(2)], dim=0), W)
    torch.testing.assert_close(
        torch.cat([shard_row(W, r, 2) for r in range(2)], dim=1), W)


def test_shard_heads_splits_on_head_boundaries():
    num_heads, head_dim, hidden = 8, 16, 128
    W = torch.randn(num_heads * head_dim, hidden)
    s0 = shard_heads(W, 0, 2, num_heads, head_dim)
    s1 = shard_heads(W, 1, 2, num_heads, head_dim)
    assert s0.shape == (4 * head_dim, hidden)
    torch.testing.assert_close(s0, W[:4 * head_dim])
    torch.testing.assert_close(s1, W[4 * head_dim:])
    torch.testing.assert_close(torch.cat([s0, s1], dim=0), W)


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
def test_sharded_mlp_matches_unsharded(world_size):
    """The only correctness bar that matters: same answer, any world size."""
    torch.manual_seed(0)
    x = torch.randn(4, 64)
    W1 = torch.randn(256, 64) / 8
    W2 = torch.randn(64, 256) / 16
    want = F.gelu(x @ W1.T) @ W2.T
    got = tp_mlp_forward(x, W1, W2, world_size)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("world_size", [2, 4])
def test_sharded_attention_matches_unsharded(world_size):
    torch.manual_seed(0)
    H, D = 8, 16
    hidden = H * D
    x = torch.randn(2, 10, hidden)
    Wq, Wk, Wv, Wo = (torch.randn(hidden, hidden) / 8 for _ in range(4))
    want = tp_attention_forward(x, Wq, Wk, Wv, Wo, 1, H, D)
    got = tp_attention_forward(x, Wq, Wk, Wv, Wo, world_size, H, D)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


def test_splitting_mid_head_is_wrong():
    """Demonstrates the classic bug, so you recognise it.

    Slicing the Q/K/V projection at an arbitrary row instead of a head
    boundary mixes different heads' subspaces. The output has the right shape
    and a plausible magnitude, and is simply incorrect.
    """
    torch.manual_seed(0)
    H, D = 4, 8
    hidden = H * D
    W = torch.randn(hidden, hidden)
    correct = shard_heads(W, 0, 2, H, D)
    wrong = W[3:3 + correct.shape[0]]            # off by 3 rows
    assert wrong.shape == correct.shape, "same shape..."
    assert not torch.allclose(wrong, correct), "...different weights"


def test_communication_accounting():
    # 4096 hidden, bf16, 32 layers, 2 all-reduces per layer
    b = allreduce_bytes_per_token(4096, 2, 32, 2)
    assert b == 4096 * 2 * 32 * 2
    print(f"\n  hidden 4096, bf16, 32 layers, 2 all-reduces/layer:")
    print(f"    {b / 1024:.0f} KB per token per rank")
    print(f"    at 4000 tok/s that is {b * 4000 / 1e9:.1f} GB/s of collective traffic")
    print("\n  \033[2mNVLink swallows that. PCIe or Ethernet does not, which is")
    print("  why tensor parallelism is an INTRA-node technique and pipeline")
    print("  parallelism is what you reach for between nodes.\033[0m")


@pytest.mark.timeout(180)
def test_real_two_rank_all_reduce():
    """Real processes, real dist.all_reduce. gloo on CPU, since you have 1 GPU.

    On multi-GPU hardware this is NCCL and the call is identical.
    """
    torch.manual_seed(0)
    x = torch.randn(4, 64)
    W1 = torch.randn(256, 64) / 8
    W2 = torch.randn(64, 256) / 16
    want = F.gelu(x @ W1.T) @ W2.T

    got = run_distributed_mlp(W1, W2, x, world_size=2)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)
    print("\n  2 spawned ranks, gloo all-reduce, output matches single-rank")
    print("  \033[2mYou have now built every major piece of a modern inference")
    print("  engine. Go read the vLLM source -- it will look familiar.\033[0m")
