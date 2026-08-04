"""Stage 20 - tensor parallelism.

`./vc lore 20` for the insight. `./vc test 20` to check yourself.

You have one GPU, so you will not get a speedup here. What you will get is the
sharding and collective logic right, and an understanding of exactly where the
communication lands -- which is the part that decides whether TP is worth it.

The pattern, and the reason it is only ONE all-reduce per block:

    MLP:  out = W2 @ act(W1 @ x)

        W1 is COLUMN-parallel   (split the output dim)  -> each rank owns a
                                                           slice of the hidden
        W2 is ROW-parallel      (split the input dim)   -> consumes that slice
                                                           and produces a
                                                           PARTIAL full output

    No communication between them. One all-reduce at the end sums the partials.

    Attention: shard by whole HEADS (Q, K, V column-parallel), output
    projection row-parallel. Again one all-reduce.

Sharding mid-head instead of by whole heads is the classic bug: it mixes parts
of different heads' subspaces, and the output looks plausible while being
completely wrong.
"""

import torch
import torch.nn.functional as F


def shard_column(W, rank, world_size):
    """W (out, in) -> (out // world_size, in). Splits the OUTPUT dimension."""
    raise NotImplementedError("stage 20: implement shard_column")


def shard_row(W, rank, world_size):
    """W (out, in) -> (out, in // world_size). Splits the INPUT dimension."""
    raise NotImplementedError("stage 20: implement shard_row")


def shard_heads(W, rank, world_size, num_heads, head_dim):
    """Split a (num_heads * head_dim, hidden) projection by whole heads."""
    raise NotImplementedError("stage 20: implement shard_heads")


def tp_mlp_forward(x, W1, W2, world_size, act=F.gelu):
    """Sharded MLP, all ranks simulated in-process.

    For each rank: take its W1 column shard and its W2 row shard, compute
    act(x @ w1.T) @ w2.T -- a partial sum of the full output. Then sum across
    ranks. That sum IS the all-reduce.

    Result must equal the unsharded MLP for every world_size.
    """
    raise NotImplementedError("stage 20: implement tp_mlp_forward")


def tp_attention_forward(x, Wq, Wk, Wv, Wo, world_size, num_heads, head_dim):
    """Sharded attention. Q/K/V split by head, output projection row-parallel.

    Each rank runs attention over its own subset of heads -- heads are already
    independent, which is exactly why this decomposition is so clean.
    """
    raise NotImplementedError("stage 20: implement tp_attention_forward")


def allreduce_bytes_per_token(hidden_size, dtype_bytes=2, num_layers=1,
                              ops_per_layer=2):
    """Bytes each rank all-reduces per token.

        hidden_size * dtype_bytes * num_layers * ops_per_layer

    Work this out before adopting TP. For a 4096-hidden 32-layer model in bf16
    it is 512 KB per token per rank, every token. Over NVLink that disappears;
    over PCIe or Ethernet it can dominate your decode step, which is why TP is
    an intra-node technique and pipeline parallelism is the inter-node one.
    """
    raise NotImplementedError("stage 20: implement allreduce_bytes_per_token")


def run_distributed_mlp(W1, W2, x, world_size=2, backend="gloo", port=29517):
    """The same MLP, but with real processes and a real dist.all_reduce.

    Spawn world_size processes, init_process_group, shard, compute the partial,
    dist.all_reduce(SUM), and return rank 0's result.

    One trap: return plain Python data from the child, not a Tensor. Tensors
    cross process boundaries as shared-memory handles, and the parent will try
    to map one after the child has exited.

    gloo on CPU is used here because you have a single GPU. On real multi-GPU
    hardware this is NCCL, and the collective is the same call.
    """
    raise NotImplementedError("stage 20: implement run_distributed_mlp")
