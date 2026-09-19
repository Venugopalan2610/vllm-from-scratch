"""Reference solution, stage 03 (jax)."""

import time

import jax
import jax.numpy as jnp

WARMUP_CALLS = 5
FILLER_TOKEN = 1000


def model_bytes(model) -> int:
    return sum(int(leaf.size) * leaf.dtype.itemsize
               for leaf in jax.tree.leaves(model.params))


def _milliseconds_per_call(call, num_calls):
    """Warm up first, so that compilation is not timed. Block one time at the
    end. A block after every call makes the dispatch serial."""
    result = None
    for _ in range(WARMUP_CALLS):
        result = call()
    jax.block_until_ready(result)

    start = time.perf_counter()
    for _ in range(num_calls):
        result = call()
    jax.block_until_ready(result)
    return (time.perf_counter() - start) / num_calls * 1000


def _filler_ids(num_tokens):
    return jnp.full((1, num_tokens), FILLER_TOKEN, dtype=jnp.int32)


def time_prefill(model, num_tokens: int, iters: int = 10) -> float:
    prompt_ids = _filler_ids(num_tokens)
    return _milliseconds_per_call(lambda: model.forward(prompt_ids)[0], iters)


def time_decode(model, context_len: int, steps: int = 40) -> float:
    cache = model.init_cache(1, context_len + steps + 8)
    _, cache = model.forward(_filler_ids(context_len), cache=cache,
                             cache_len=jnp.zeros(1, jnp.int32))
    one_token = _filler_ids(1)
    state = {"cache": cache, "cache_len": context_len}

    def decode_step():
        position = state["cache_len"]
        logits, state["cache"] = model.forward(
            one_token, positions=jnp.asarray([[position]], jnp.int32),
            cache=state["cache"],
            cache_len=jnp.asarray([position], jnp.int32))
        state["cache_len"] = position + 1
        return logits

    return _milliseconds_per_call(decode_step, steps)


def achieved_gbs(num_bytes: int, ms_per_token: float) -> float:
    return num_bytes / (ms_per_token / 1000) / 1e9
