"""Reference solution, stage 20 - tensor parallelism."""

import torch
import torch.nn.functional as F


def shard_column(W, rank, world_size):
    """W (out, in) -> (out // world_size, in). Splits the OUTPUT dimension."""
    assert W.shape[0] % world_size == 0
    n = W.shape[0] // world_size
    return W[rank * n:(rank + 1) * n, :]


def shard_row(W, rank, world_size):
    """W (out, in) -> (out, in // world_size). Splits the INPUT dimension."""
    assert W.shape[1] % world_size == 0
    n = W.shape[1] // world_size
    return W[:, rank * n:(rank + 1) * n]


def shard_heads(W, rank, world_size, num_heads, head_dim):
    """Split a (num_heads*head_dim, hidden) projection by whole HEADS.

    Splitting mid-head would mix parts of different heads' subspaces and
    silently produce wrong attention.
    """
    assert num_heads % world_size == 0
    per = num_heads // world_size
    lo = rank * per * head_dim
    hi = (rank + 1) * per * head_dim
    return W[lo:hi, :]


def tp_mlp_forward(x, W1, W2, world_size, act=F.gelu):
    """MLP = W2 @ act(W1 @ x), sharded across world_size ranks.

    W1 column-parallel, W2 row-parallel. Each rank produces a PARTIAL sum of
    the full output, and one all-reduce finishes it. Crucially there is NO
    communication between the two matmuls -- that pairing is the whole trick.
    """
    partials = []
    for r in range(world_size):
        w1 = shard_column(W1, r, world_size)
        w2 = shard_row(W2, r, world_size)
        h = act(x @ w1.T)          # (B, inter/ws)  -- rank-local
        partials.append(h @ w2.T)  # (B, out)       -- partial sum
    return torch.stack(partials).sum(0)     # the all-reduce


def tp_attention_forward(x, Wq, Wk, Wv, Wo, world_size, num_heads, head_dim):
    """QKV projections are column-parallel by head; the output projection is
    row-parallel. Again exactly one all-reduce."""
    partials = []
    per = num_heads // world_size
    for r in range(world_size):
        wq = shard_heads(Wq, r, world_size, num_heads, head_dim)
        wk = shard_heads(Wk, r, world_size, num_heads, head_dim)
        wv = shard_heads(Wv, r, world_size, num_heads, head_dim)
        wo = shard_row(Wo, r, world_size)

        B, T, _ = x.shape
        q = (x @ wq.T).view(B, T, per, head_dim).transpose(1, 2)
        k = (x @ wk.T).view(B, T, per, head_dim).transpose(1, 2)
        v = (x @ wv.T).view(B, T, per, head_dim).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1, 2).reshape(B, T, per * head_dim)
        partials.append(o @ wo.T)
    return torch.stack(partials).sum(0)


def allreduce_bytes_per_token(hidden_size, dtype_bytes=2, num_layers=1,
                              ops_per_layer=2):
    """Bytes each rank contributes to all-reduces, per token.

    One all-reduce per attention block and one per MLP block, each over a
    hidden_size vector.
    """
    return hidden_size * dtype_bytes * num_layers * ops_per_layer


# ---- real distributed (2 ranks) -------------------------------------

def _dist_worker(rank, world_size, W1, W2, x, q, backend, port):
    import os
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    w1 = shard_column(W1, rank, world_size)
    w2 = shard_row(W2, rank, world_size)
    h = F.gelu(x @ w1.T)
    partial = h @ w2.T

    dist.all_reduce(partial, op=dist.ReduceOp.SUM)
    if rank == 0:
        # Send plain Python data, not a Tensor. A tensor crosses process
        # boundaries as a shared-memory handle, and the parent would try to
        # map it after this process has already exited.
        q.put(partial.detach().cpu().tolist())
    dist.destroy_process_group()


def run_distributed_mlp(W1, W2, x, world_size=2, backend="gloo", port=29517):
    """Run the sharded MLP across real processes with real collectives."""
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    mp.start_processes(
        _dist_worker,
        args=(world_size, W1, W2, x, q, backend, port),
        nprocs=world_size, start_method="spawn", join=True,
    )
    return torch.tensor(q.get())
