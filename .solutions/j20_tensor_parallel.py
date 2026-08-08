"""Reference solution, stage 20 (jax) - tensor parallelism with shard_map."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

shard_map = getattr(jax, "shard_map", None)
if shard_map is None:                                    # older jax
    from jax.experimental.shard_map import shard_map


def make_mesh(world_size=2, axis="tp"):
    """A mesh of `world_size` CPU devices.

    One GPU is one device, so the ranks come from the CPU backend --
    XLA_FLAGS=--xla_force_host_platform_device_count=8, which jvllm sets on
    import. These are real devices with real shardings and a real collective,
    not a Python loop pretending to be ranks.
    """
    devices = jax.devices("cpu")
    if len(devices) < world_size:
        raise ValueError(
            f"need {world_size} CPU devices, have {len(devices)}. Set "
            "XLA_FLAGS=--xla_force_host_platform_device_count=8 before jax "
            "initialises."
        )
    return Mesh(np.asarray(devices[:world_size]), (axis,))


# ---- what each rank owns --------------------------------------------

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
    return W[rank * per * head_dim:(rank + 1) * per * head_dim, :]


# ---- the sharded blocks ---------------------------------------------

@functools.partial(jax.jit, static_argnums=(3, 4))
def _tp_mlp(x, W1, W2, mesh, axis):
    def body(x, w1, w2):
        h = jax.nn.gelu(x @ w1.T)          # (B, inter/ws)  -- rank-local
        return jax.lax.psum(h @ w2.T, axis)  # (B, out) -- one all-reduce

    return shard_map(
        body, mesh=mesh,
        in_specs=(P(), P(axis, None), P(None, axis)),
        out_specs=P(),
    )(x, W1, W2)


def tp_mlp_forward(x, W1, W2, mesh, axis="tp"):
    """MLP = W2 @ act(W1 @ x), sharded across the mesh.

    W1 column-parallel, W2 row-parallel. Each rank produces a PARTIAL sum of
    the full output and one psum finishes it. Crucially there is NO
    communication between the two matmuls -- that pairing is the whole trick,
    and it is what the two PartitionSpecs are saying.
    """
    x, W1, W2 = _place(mesh, axis, x, W1, W2)
    return _tp_mlp(x, W1, W2, mesh, axis)


@functools.partial(jax.jit, static_argnums=(5, 6, 7, 8))
def _tp_attention(x, Wq, Wk, Wv, Wo, mesh, axis, num_heads, head_dim):
    def body(x, wq, wk, wv, wo):
        B, T, _ = x.shape
        per = wq.shape[0] // head_dim

        def heads(w):
            return (x @ w.T).reshape(B, T, per, head_dim).transpose(0, 2, 1, 3)

        q, k, v = heads(wq), heads(wk), heads(wv)
        scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(head_dim)
        causal = jnp.tril(jnp.ones((T, T), bool))
        scores = jnp.where(causal, scores, jnp.finfo(scores.dtype).min)
        o = jax.nn.softmax(scores, axis=-1) @ v
        o = o.transpose(0, 2, 1, 3).reshape(B, T, per * head_dim)
        return jax.lax.psum(o @ wo.T, axis)

    return shard_map(
        body, mesh=mesh,
        in_specs=(P(), P(axis, None), P(axis, None), P(axis, None),
                  P(None, axis)),
        out_specs=P(),
    )(x, Wq, Wk, Wv, Wo)


def tp_attention_forward(x, Wq, Wk, Wv, Wo, mesh, num_heads, head_dim,
                         axis="tp"):
    """QKV column-parallel by head, output projection row-parallel.

    Again exactly one all-reduce: attention is head-local, so no rank needs
    anybody else's heads until the output projection has already been summed
    into the shape of the residual stream.
    """
    x, Wq, Wk, Wv, Wo = _place(mesh, axis, x, Wq, Wk, Wv, Wo, attn=True)
    return _tp_attention(x, Wq, Wk, Wv, Wo, mesh, axis, num_heads, head_dim)


def _place(mesh, axis, x, *weights, attn=False):
    """Commit the arrays to the mesh's devices with the right sharding.

    Without this the inputs live on whatever device made them (the GPU), and
    shard_map has nothing to shard. Doing it explicitly is also the clearest
    statement of the whole scheme: two of these specs ARE tensor parallelism.
    """
    def put(a, spec):
        return jax.device_put(a, NamedSharding(mesh, spec))

    specs = ([P(axis, None)] * 3 + [P(None, axis)]) if attn \
        else [P(axis, None), P(None, axis)]
    return (put(x, P()), *[put(w, s) for w, s in zip(weights, specs)])


def allreduce_bytes_per_token(hidden_size, dtype_bytes=2, num_layers=1,
                              ops_per_layer=2):
    """Bytes each rank contributes to all-reduces, per token.

    One all-reduce per attention block and one per MLP block, each over a
    hidden_size vector.
    """
    return hidden_size * dtype_bytes * num_layers * ops_per_layer
