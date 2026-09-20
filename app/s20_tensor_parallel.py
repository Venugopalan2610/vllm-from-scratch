"""Stage 20 - tensor parallelism.

`./vc lore 20` for the insight. `./vc test 20` to check yourself.

You have one GPU, so you get no speedup here. You get two other things. You
make the sharding and the collective logic correct, and you learn exactly
where the communication occurs. That second part decides if TP is worth its
cost.

**Scope.** This stage teaches the SHAPE of tensor parallelism: which
weights split, where the all-reduce goes, and how many bytes it moves.
It does NOT teach the system: NCCL, overlapping communication with
compute, sequence parallelism for the norms, and non-deterministic
reductions. If you have two GPUs, the optional exercise at the end runs
real NCCL. Otherwise, know that the algebra is correct and the
engineering is a separate project.

The pattern, and the reason for only ONE all-reduce in each block:

    MLP:  out = W2 @ act(W1 @ x)

        W1 is COLUMN-parallel   (split the output dim)  -> each rank owns a
                                                           part of the hidden
        W2 is ROW-parallel      (split the input dim)   -> uses that part and
                                                           makes a PARTIAL
                                                           full output

    No communication between them. One all-reduce at the end adds the
    partial outputs.

    Attention: shard on whole HEADS (Q, K, V column-parallel), and the output
    projection row-parallel. Again one all-reduce.

The usual bug is a shard in the middle of a head, not a shard on whole
heads. It mixes parts of different heads. The output then looks correct and
is completely wrong.
"""

import torch
import torch.nn.functional as F


def shard_column(weight, rank, world_size):
    """weight (out, in) -> (out // world_size, in). It splits the OUTPUT
    dimension."""
    raise NotImplementedError("stage 20: implement shard_column")


def shard_row(weight, rank, world_size):
    """weight (out, in) -> (out, in // world_size). It splits the INPUT
    dimension."""
    raise NotImplementedError("stage 20: implement shard_row")


def shard_heads(weight, rank, world_size, num_heads, head_dim):
    """Split a (num_heads * head_dim, hidden) projection on whole heads."""
    raise NotImplementedError("stage 20: implement shard_heads")


def tp_mlp_forward(inputs, up_weight, down_weight, world_size, act=F.gelu):
    """The sharded MLP. All ranks run in this process.

    For each rank: take its column shard of up_weight and its row shard of
    down_weight, and compute act(inputs @ up.T) @ down.T. That is a partial
    sum of the output. Then add the partial sums of all ranks. That sum IS
    the all-reduce.

    The result must equal the MLP with no shards, for each world_size.
    """
    raise NotImplementedError("stage 20: implement tp_mlp_forward")


def tp_attention_forward(inputs, query_weight, key_weight, value_weight,
                         output_weight, world_size, num_heads, head_dim):
    """Sharded attention. Q, K and V split on heads. The output projection is
    row-parallel.

    Each rank runs attention over its own heads. The heads are already
    independent, and that is why this split is so clean.
    """
    raise NotImplementedError("stage 20: implement tp_attention_forward")


def allreduce_bytes_per_token(hidden_size, dtype_bytes=2, num_layers=1,
                              ops_per_layer=2):
    """The bytes that each rank all-reduces for each token.

        hidden_size * dtype_bytes * num_layers * ops_per_layer

    Calculate this before you use TP. For a model with a hidden size of 4096
    and 32 layers in bf16, it is 512 KB for each token on each rank. Over
    NVLink that cost is small. Over PCIe or Ethernet it can be most of the
    decode step. That is why people use TP inside one node, and pipeline
    parallelism across nodes.
    """
    raise NotImplementedError("stage 20: implement allreduce_bytes_per_token")


def run_distributed_mlp(up_weight, down_weight, inputs, world_size=2,
                        backend="gloo", port=29517):
    """The same MLP, with real processes and a real dist.all_reduce.

    Start world_size processes, call init_process_group, shard, compute the
    partial sum, call dist.all_reduce(SUM), and return the result of rank 0.

    A trap: return Python data from the child process, not a Tensor. A tensor
    goes to another process as a shared-memory handle, and the parent then
    tries to map it after the child stops.

    This uses gloo on the CPU, because you have one GPU. On real hardware
    with many GPUs this is NCCL, and the collective is the same call.
    """
    raise NotImplementedError("stage 20: implement run_distributed_mlp")
