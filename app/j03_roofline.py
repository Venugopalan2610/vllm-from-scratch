"""Stage 03 (JAX) - prefill vs decode: two different machines.

`./vc lore 3 --jax` for the insight. `./vc test 3 --jax` to check yourself.

WHAT YOU'RE BUILDING

    model_bytes(model) -> int                          total weight bytes
    time_prefill(model, n_tokens, iters=10) -> float    ms per FORWARD PASS
    time_decode(model, ctx_len, steps=40) -> float      ms per TOKEN
    achieved_gbs(nbytes, ms_per_token) -> float

This stage produces no new capability. It produces a NUMBER you will spend the
next seventeen stages moving. Do not skip it.

THE ONE THING THAT WILL RUIN YOUR MEASUREMENT

JAX dispatch is ASYNCHRONOUS. `model.forward(ids)` returns as soon as the work
is enqueued, not when it is finished. Time a loop without blocking and you
measure how fast Python can queue work -- which on a 0.6B model reads as
0.05 ms/token, an impossible 20 TB/s of memory bandwidth, and a decode step
apparently faster than the roofline allows.

    out = fn()
    jax.block_until_ready(out)      # <- the whole measurement depends on this

Block ONCE at the end of the timed loop, not inside it: blocking every
iteration serialises dispatch and measures a different (also wrong) thing.

And warm up separately. The first call on a new shape compiles, and XLA
compilation is seconds. Run the warmups, block, THEN start the clock.

    model.params        the pytree. jax.tree.leaves() to walk it.
    x.size, x.dtype.itemsize   bytes of one array

FOR time_decode

Steady-state decode, not prefill: preallocate a cache, prefill the context
once, then step one token at a time with the shapes held constant. cache_len
changes VALUE every step but not SHAPE, so the compiled step is reused -- if
your loop recompiles, you are timing the compiler again.
"""


def model_bytes(model) -> int:
    """Total bytes of the parameter pytree."""
    raise NotImplementedError("stage 03 (jax): implement model_bytes")


def time_prefill(model, n_tokens: int, iters: int = 10) -> float:
    """Milliseconds for one forward pass over n_tokens."""
    raise NotImplementedError("stage 03 (jax): implement time_prefill")


def time_decode(model, ctx_len: int, steps: int = 40) -> float:
    """Milliseconds per token in steady-state decode at a given context."""
    raise NotImplementedError("stage 03 (jax): implement time_decode")


def achieved_gbs(nbytes: int, ms_per_token: float) -> float:
    """Effective memory bandwidth, in GB/s, for reading nbytes per token."""
    raise NotImplementedError("stage 03 (jax): implement achieved_gbs")
