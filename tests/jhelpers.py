"""Shared fixtures for the JAX track. The counterpart of tests/helpers.py.

Two things here are worth reading before you use them:

`jbench_ms` blocks. JAX dispatch is asynchronous -- `f(x)` returns as soon as
the work is ENQUEUED, not when it is done. A timing loop without
block_until_ready() measures Python, and on a small model that reads as a 50x
speedup you did not get. Every measurement in this track goes through here.

`build_paged` shuffles the physical blocks, exactly as the torch helper does. A
correct paged kernel cannot care what order the blocks landed in.
"""

import math
import random
import time

import jax
import jax.numpy as jnp
import numpy as np


def build_paged(keys, values, block_size, shuffle=True, seed=0):
    """Scatter dense (S, KVH, L, D) K/V into a paged cache.

    Returns (key_cache, value_cache, block_tables, context_lens) with

        key_cache     (num_blocks, kv_heads, block_size, head_dim)
        block_tables  (S, blocks_per_seq) int32
        context_lens  (S,) int32

    Same layout as the torch track's helper, so the two stage-07 oracles are
    comparable number for number.
    """
    S, KVH, L, D = keys.shape
    nb = math.ceil(L / block_size)
    ids = list(range(S * nb))
    if shuffle:
        random.Random(seed).shuffle(ids)

    kc = np.zeros((S * nb, KVH, block_size, D), dtype=np.asarray(keys).dtype)
    vc = np.zeros_like(kc)
    bt = np.zeros((S, nb), dtype=np.int32)
    k_np, v_np = np.asarray(keys), np.asarray(values)

    for s in range(S):
        for b in range(nb):
            phys = ids[s * nb + b]
            bt[s, b] = phys
            lo, hi = b * block_size, min((b + 1) * block_size, L)
            kc[phys, :, : hi - lo] = k_np[s, :, lo:hi]
            vc[phys, :, : hi - lo] = v_np[s, :, lo:hi]

    return (jnp.asarray(kc), jnp.asarray(vc), jnp.asarray(bt),
            jnp.full((S,), L, dtype=jnp.int32))


def rand_kv(S, KVH, L, D, dtype=jnp.float32, seed=1234):
    k1, k2 = jax.random.split(jax.random.key(seed))
    shape = (S, KVH, L, D)
    return (jax.random.normal(k1, shape, dtype=dtype),
            jax.random.normal(k2, shape, dtype=dtype))


def jbench_ms(fn, iters=30, warmup=5):
    """Milliseconds per call, with the device actually waited on.

    The warmup runs also absorb XLA compilation, which is seconds and would
    otherwise land entirely in the first timed iteration.
    """
    for _ in range(warmup):
        out = fn()
    jax.block_until_ready(out)

    t = time.perf_counter()
    for _ in range(iters):
        out = fn()
    jax.block_until_ready(out)
    return (time.perf_counter() - t) / iters * 1000


def cache_size(jitted):
    """How many shape signatures a jitted function has compiled for.

    This is the number stage 12 lives and dies by. `jax.jit(f)._cache_size()`
    goes up by one every time a call arrives with a shape combination XLA has
    not seen; it stays flat on a cache hit. A decode loop whose cache size
    climbs with every step is recompiling per step, which is the JAX version of
    paying kernel-launch overhead per step -- only about a thousand times
    worse.
    """
    return jitted._cache_size()
