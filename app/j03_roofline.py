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

JAX dispatch is ASYNCHRONOUS. `model.forward(ids)` returns when the work
reaches the queue, and not when the device finishes it.

Time a loop with no block, and you measure how fast Python fills that queue.
On a 0.6B model that reads as 0.05 ms/token. That is 20 TB/s of memory
bandwidth, which is impossible, and a decode step faster than the roofline
permits.

    out = fn()
    jax.block_until_ready(out)      # <- the whole measurement depends on this

Block ONCE at the end of the timed loop, and not inside it. A block at every
iteration serialises the dispatch, and it measures a different wrong thing.

And warm up separately. The first call on a new shape compiles, and XLA
compilation is seconds. Run the warmups, block, THEN start the clock.

    model.params        the pytree. jax.tree.leaves() to walk it.
    x.size, x.dtype.itemsize   bytes of one array

FOR time_decode

Measure steady-state decode, and not prefill. Preallocate a cache, prefill the
context one time, then step one token at a time and hold the shapes constant.

cache_len changes its VALUE at every step, and it never changes SHAPE. So XLA
reuses the compiled step. If your loop recompiles, you time the compiler
again.
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
