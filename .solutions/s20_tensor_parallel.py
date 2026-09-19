"""Reference solution, stage 20 - tensor parallelism."""

import torch
import torch.nn.functional as F


def shard_column(weight, rank, world_size):
    """weight (out, in) -> (out // world_size, in). It splits the OUTPUT
    dimension."""
    assert weight.shape[0] % world_size == 0
    rows_per_rank = weight.shape[0] // world_size
    return weight[rank * rows_per_rank:(rank + 1) * rows_per_rank, :]


def shard_row(weight, rank, world_size):
    """weight (out, in) -> (out, in // world_size). It splits the INPUT
    dimension."""
    assert weight.shape[1] % world_size == 0
    columns_per_rank = weight.shape[1] // world_size
    return weight[:, rank * columns_per_rank:(rank + 1) * columns_per_rank]


def shard_heads(weight, rank, world_size, num_heads, head_dim):
    """Split a (num_heads * head_dim, hidden) projection on whole HEADS.

    A split inside a head mixes the parts of two heads, and the attention is
    then wrong with no error.
    """
    assert num_heads % world_size == 0
    rows_per_rank = num_heads // world_size * head_dim
    return weight[rank * rows_per_rank:(rank + 1) * rows_per_rank, :]


def all_reduce(partials):
    """What one all-reduce computes: the sum of the partial outputs."""
    return torch.stack(partials).sum(0)


def mlp_partial(inputs, up_weight, down_weight, rank, world_size, act):
    """The part of the MLP output that one rank computes."""
    hidden = act(inputs @ shard_column(up_weight, rank, world_size).T)
    return hidden @ shard_row(down_weight, rank, world_size).T


def tp_mlp_forward(inputs, up_weight, down_weight, world_size, act=F.gelu):
    """MLP = down @ act(up @ inputs), sharded across world_size ranks.

    The up weight is column-parallel and the down weight is row-parallel. Each rank makes a PARTIAL
    sum of the output, and one all-reduce completes it. There is NO
    communication between the two matmuls. That pair is the method.
    """
    return all_reduce([mlp_partial(inputs, up_weight, down_weight, rank,
                                   world_size, act)
                       for rank in range(world_size)])


def attention_partial(inputs, projections, rank, world_size, num_heads,
                      head_dim):
    """The part of the attention output that one rank computes."""
    query_weight, key_weight, value_weight, output_weight = projections
    batch, seq_len, _ = inputs.shape
    heads_per_rank = num_heads // world_size

    def local_heads(weight):
        shard = shard_heads(weight, rank, world_size, num_heads, head_dim)
        return (inputs @ shard.T).view(batch, seq_len, heads_per_rank,
                                       head_dim).transpose(1, 2)

    attended = F.scaled_dot_product_attention(
        local_heads(query_weight), local_heads(key_weight),
        local_heads(value_weight), is_causal=True)
    attended = attended.transpose(1, 2).reshape(batch, seq_len,
                                                heads_per_rank * head_dim)
    return attended @ shard_row(output_weight, rank, world_size).T


def tp_attention_forward(inputs, query_weight, key_weight, value_weight,
                         output_weight, world_size, num_heads, head_dim):
    """The QKV projections are column-parallel on heads. The output
    projection is row-parallel. Again, one all-reduce."""
    projections = (query_weight, key_weight, value_weight, output_weight)
    return all_reduce([attention_partial(inputs, projections, rank, world_size,
                                         num_heads, head_dim)
                       for rank in range(world_size)])


def allreduce_bytes_per_token(hidden_size, dtype_bytes=2, num_layers=1,
                              ops_per_layer=2):
    """The bytes that each rank gives to the all-reduces, for each token.

    One all-reduce for each attention block and one for each MLP block, each
    over a vector of hidden_size.
    """
    return hidden_size * dtype_bytes * num_layers * ops_per_layer


# ---- real distributed (2 ranks) -------------------------------------

def _distributed_worker(rank, world_size, up_weight, down_weight, inputs,
                        result_queue, backend, port):
    import os

    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    partial = mlp_partial(inputs, up_weight, down_weight, rank, world_size,
                          F.gelu)
    dist.all_reduce(partial, op=dist.ReduceOp.SUM)
    if rank == 0:
        # Send Python data, not a Tensor. A tensor goes to another process
        # as a shared-memory handle, and the parent then tries to map it
        # after this process stops.
        result_queue.put(partial.detach().cpu().tolist())
    dist.destroy_process_group()


def run_distributed_mlp(up_weight, down_weight, inputs, world_size=2, backend="gloo", port=29517):
    """Run the sharded MLP in real processes, with real collectives."""
    import torch.multiprocessing as mp

    result_queue = mp.get_context("spawn").Queue()
    mp.start_processes(
        _distributed_worker,
        args=(world_size, up_weight, down_weight, inputs, result_queue,
              backend, port),
        nprocs=world_size, start_method="spawn", join=True)
    return torch.tensor(result_queue.get())
