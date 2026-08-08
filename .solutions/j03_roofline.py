"""Reference solution, stage 03 (jax)."""

import time

import jax
import jax.numpy as jnp


def model_bytes(model) -> int:
    return sum(int(x.size) * x.dtype.itemsize
               for x in jax.tree.leaves(model.params))


def _timed(fn, iters, warmup=5):
    """Warm up (absorbing compilation), block, then time. Block ONCE at the
    end -- blocking every iteration serialises dispatch."""
    out = None
    for _ in range(warmup):
        out = fn()
    jax.block_until_ready(out)

    t = time.perf_counter()
    for _ in range(iters):
        out = fn()
    jax.block_until_ready(out)
    return (time.perf_counter() - t) / iters * 1000


def time_prefill(model, n_tokens: int, iters: int = 10) -> float:
    ids = jnp.full((1, n_tokens), 1000, dtype=jnp.int32)
    return _timed(lambda: model.forward(ids)[0], iters)


def time_decode(model, ctx_len: int, steps: int = 40) -> float:
    max_len = ctx_len + steps + 8
    cache = model.init_cache(1, max_len)
    ids = jnp.full((1, ctx_len), 1000, dtype=jnp.int32)
    _, cache = model.forward(ids, cache=cache,
                             cache_len=jnp.zeros(1, jnp.int32))

    one = jnp.full((1, 1), 1000, dtype=jnp.int32)
    state = {"cache": cache, "len": ctx_len}

    def step():
        n = state["len"]
        logits, state["cache"] = model.forward(
            one, positions=jnp.asarray([[n]], jnp.int32),
            cache=state["cache"], cache_len=jnp.asarray([n], jnp.int32),
        )
        state["len"] = n + 1
        return logits

    return _timed(step, steps)


def achieved_gbs(nbytes: int, ms_per_token: float) -> float:
    return nbytes / (ms_per_token / 1000) / 1e9
