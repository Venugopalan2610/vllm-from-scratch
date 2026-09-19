"""Reference solution, stage 20 (jax) - tensor parallelism with shard_map."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

shard_map = getattr(jax, "shard_map", None)
if shard_map is None:                                    # an older jax
    from jax.experimental.shard_map import shard_map


def make_mesh(world_size=2, axis="tp"):
    """A mesh of `world_size` CPU devices.

    One GPU is one device, so the ranks come from the CPU backend:
    XLA_FLAGS=--xla_force_host_platform_device_count=8, which jvllm sets on
    import. These are real devices, with real shardings and a real
    collective. They are not a Python loop that acts as ranks.
    """
    devices = jax.devices("cpu")
    if len(devices) < world_size:
        raise ValueError(
            f"need {world_size} CPU devices, have {len(devices)}. Set "
            "XLA_FLAGS=--xla_force_host_platform_device_count=8 before jax "
            "starts.")
    return Mesh(np.asarray(devices[:world_size]), (axis,))


# ---- what each rank owns --------------------------------------------

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


# ---- the placement --------------------------------------------------

COLUMN_PARALLEL = "column"
ROW_PARALLEL = "row"


def place(mesh, axis, array, layout=None):
    """Put the array on the mesh devices, with its sharding.

    Without this, an input stays on the device that made it (the GPU), and
    shard_map has nothing to shard. These two specs ARE tensor parallelism.
    """
    spec = {None: P(), COLUMN_PARALLEL: P(axis, None),
            ROW_PARALLEL: P(None, axis)}[layout]
    return jax.device_put(array, NamedSharding(mesh, spec))


# ---- the sharded blocks ---------------------------------------------

@functools.partial(jax.jit, static_argnums=(3, 4))
def _tp_mlp(inputs, up_weight, down_weight, mesh, axis):
    def rank_body(inputs, up_shard, down_shard):
        hidden = jax.nn.gelu(inputs @ up_shard.T)       # local to the rank
        return jax.lax.psum(hidden @ down_shard.T, axis)  # one all-reduce

    return shard_map(rank_body, mesh=mesh,
                     in_specs=(P(), P(axis, None), P(None, axis)),
                     out_specs=P())(inputs, up_weight, down_weight)


def tp_mlp_forward(inputs, up_weight, down_weight, mesh, axis="tp"):
    """MLP = down @ act(up @ inputs), sharded across the mesh.

    The up weight is column-parallel and the down weight is row-parallel.
    Each rank makes a PARTIAL sum of the output, and one psum completes it.
    There is NO communication between the two matmuls. That pair is the
    method, and the two PartitionSpecs say it.
    """
    return _tp_mlp(place(mesh, axis, inputs),
                   place(mesh, axis, up_weight, COLUMN_PARALLEL),
                   place(mesh, axis, down_weight, ROW_PARALLEL), mesh, axis)


def causal_attention(query, key, value, head_dim):
    """(batch, heads, seq, head_dim) -> the same shape."""
    seq_len = query.shape[2]
    scores = query @ key.transpose(0, 1, 3, 2) / np.sqrt(head_dim)
    causal = jnp.tril(jnp.ones((seq_len, seq_len), bool))
    scores = jnp.where(causal, scores, jnp.finfo(scores.dtype).min)
    return jax.nn.softmax(scores, axis=-1) @ value


@functools.partial(jax.jit, static_argnums=(5, 6, 7, 8))
def _tp_attention(inputs, query_weight, key_weight, value_weight,
                  output_weight, mesh, axis, num_heads, head_dim):
    def rank_body(inputs, query_shard, key_shard, value_shard, output_shard):
        batch, seq_len, _ = inputs.shape
        heads_per_rank = query_shard.shape[0] // head_dim

        def local_heads(shard):
            return (inputs @ shard.T).reshape(
                batch, seq_len, heads_per_rank, head_dim).transpose(0, 2, 1, 3)

        attended = causal_attention(local_heads(query_shard),
                                    local_heads(key_shard),
                                    local_heads(value_shard), head_dim)
        attended = attended.transpose(0, 2, 1, 3).reshape(
            batch, seq_len, heads_per_rank * head_dim)
        return jax.lax.psum(attended @ output_shard.T, axis)

    column = P(axis, None)
    return shard_map(rank_body, mesh=mesh,
                     in_specs=(P(), column, column, column, P(None, axis)),
                     out_specs=P())(inputs, query_weight, key_weight,
                                    value_weight, output_weight)


def tp_attention_forward(inputs, query_weight, key_weight, value_weight,
                         output_weight, mesh, num_heads, head_dim, axis="tp"):
    """QKV are column-parallel on heads. The output projection is
    row-parallel.

    Again, one all-reduce. Attention is local to a head, so no rank needs the
    heads of another rank. The output projection then sums them into the
    shape of the residual stream.
    """
    def column(weight):
        return place(mesh, axis, weight, COLUMN_PARALLEL)

    return _tp_attention(place(mesh, axis, inputs), column(query_weight),
                         column(key_weight), column(value_weight),
                         place(mesh, axis, output_weight, ROW_PARALLEL),
                         mesh, axis, num_heads, head_dim)


def allreduce_bytes_per_token(hidden_size, dtype_bytes=2, num_layers=1,
                              ops_per_layer=2):
    """The bytes that each rank gives to the all-reduces, for each token.

    One all-reduce for each attention block and one for each MLP block, each
    over a vector of hidden_size.
    """
    return hidden_size * dtype_bytes * num_layers * ops_per_layer
