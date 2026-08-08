"""Stage 20 (JAX) - tensor parallelism with shard_map.

`./vc lore 20 --jax` for the insight. `./vc test 20 --jax` to check yourself.

Shard attention heads and MLP columns across ranks; one all-reduce per block.
The lesson is where the communication lands, not the speedup -- you have one
GPU and there is no speedup to be had.

The torch track fakes the ranks with a Python loop and then runs two real gloo
processes to prove the collective works. You do not have to fake anything. XLA
will hand you as many CPU devices as you ask for, so you get a real mesh, real
shardings and a real psum, and the code you write here is the code you would
deploy on eight accelerators.

WHAT YOU'RE BUILDING

    make_mesh(world_size=2, axis="tp") -> jax.sharding.Mesh

    shard_column(W, rank, world_size)      W (out, in) -> (out/ws, in)
    shard_row(W, rank, world_size)         W (out, in) -> (out, in/ws)
    shard_heads(W, rank, world_size, num_heads, head_dim)

    tp_mlp_forward(x, W1, W2, mesh, axis="tp")
    tp_attention_forward(x, Wq, Wk, Wv, Wo, mesh, num_heads, head_dim,
                         axis="tp")
    allreduce_bytes_per_token(hidden_size, dtype_bytes=2, num_layers=1,
                              ops_per_layer=2)

Both forwards must return exactly what the unsharded computation returns.

THE WHOLE SCHEME, IN TWO PARTITION SPECS

    MLP = W2 @ act(W1 @ x)

    W1 is COLUMN-parallel: split its output dimension.      P(axis, None)
    W2 is ROW-parallel:    split its input dimension.       P(None, axis)

Every rank computes act(W1_r @ x) entirely on its own -- that is why the split
dimensions have to pair up this way. Then each rank's W2_r @ h_r is a PARTIAL
SUM of the full output, and one psum finishes it.

There is NO communication between the two matmuls. That is the trick, and it
is the only reason tensor parallelism is affordable at all.

    shard_map(body, mesh=mesh,
              in_specs=(P(), P(axis, None), P(None, axis)),
              out_specs=P())

Inside `body` you are writing RANK-LOCAL code: `w1` is already just this rank's
slice, `x` is the full replicated input, and `jax.lax.psum(y, axis)` is the
all-reduce. What comes out is replicated again, which is exactly the shape the
residual stream needs.

ATTENTION IS THE SAME SHAPE

QKV projections are column-parallel BY WHOLE HEADS, the output projection is
row-parallel, one psum. Attention is head-local, so no rank ever needs another
rank's heads -- splitting mid-head would mix parts of different heads'
subspaces and produce attention that is wrong rather than merely approximate.

PUT THE ARRAYS ON THE MESH

An array made with jnp lives on the default device, which here is the GPU. The
mesh is CPU devices. Commit them first or shard_map has nothing to shard:

    jax.device_put(W1, NamedSharding(mesh, P(axis, None)))

TRAPS

  - `num_heads % world_size == 0` and `hidden % world_size == 0`. Assert it; an
    uneven split fails deep inside shard_map with a shape error that says
    nothing about heads.

  - Exactly ONE psum per block. Two means you all-reduced something that was
    already replicated, and on real hardware that is the difference between
    tensor parallelism being cheap and being pointless. It is invisible in the
    numbers -- the answer stays correct, it just costs twice the interconnect.
    `jax.jit(f).lower(...).as_text()` and count `all_reduce`.

  - The Mesh is hashable, so it can be a static jit argument. The device_put
    cannot happen inside jit -- do it outside.
"""


def make_mesh(world_size=2, axis="tp"):
    """A mesh of `world_size` CPU devices."""
    raise NotImplementedError("stage 20 (jax): implement make_mesh")


def shard_column(W, rank, world_size):
    """Split the OUTPUT dimension."""
    raise NotImplementedError("stage 20 (jax): implement shard_column")


def shard_row(W, rank, world_size):
    """Split the INPUT dimension."""
    raise NotImplementedError("stage 20 (jax): implement shard_row")


def shard_heads(W, rank, world_size, num_heads, head_dim):
    """Split a (num_heads*head_dim, hidden) projection by whole heads."""
    raise NotImplementedError("stage 20 (jax): implement shard_heads")


def tp_mlp_forward(x, W1, W2, mesh, axis="tp"):
    """Column-parallel then row-parallel, one psum."""
    raise NotImplementedError("stage 20 (jax): implement tp_mlp_forward")


def tp_attention_forward(x, Wq, Wk, Wv, Wo, mesh, num_heads, head_dim,
                         axis="tp"):
    """QKV column-parallel by head, output projection row-parallel, one psum."""
    raise NotImplementedError("stage 20 (jax): implement tp_attention_forward")


def allreduce_bytes_per_token(hidden_size, dtype_bytes=2, num_layers=1,
                              ops_per_layer=2):
    """Bytes each rank contributes to all-reduces, per token."""
    raise NotImplementedError(
        "stage 20 (jax): implement allreduce_bytes_per_token")
