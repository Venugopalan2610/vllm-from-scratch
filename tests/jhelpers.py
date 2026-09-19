"""Shared helpers for the JAX track. The JAX twin of tests/helpers.py.

Read two things here before you use them:

`jbench_ms` blocks. JAX dispatch is asynchronous. `f(x)` returns when the
work gets to the queue, not when the device finishes it. A timing loop with
no block_until_ready() measures Python. On a small model, that reads as a
50x speedup that you did not get. Every measurement of this track uses this
function.

`build_paged` shuffles the physical blocks, as the torch helper does. A
correct paged kernel must not depend on the order of the blocks.
"""

import math
import time

import jax
import jax.numpy as jnp
import numpy as np

from tests.helpers import shuffled_block_ids


def build_paged(keys, values, block_size, shuffle=True, seed=0):
    """Scatter dense (seqs, kv_heads, context, D) K and V into a paged cache.

    -> (key_cache, value_cache, block_tables, context_lens), with

        key_cache     (num_blocks, kv_heads, block_size, head_dim)
        block_tables  (seqs, blocks_per_seq) int32
        context_lens  (seqs,) int32

    The layout is the layout of the torch helper, so the two stage-07
    oracles give the same numbers.
    """
    num_seqs, num_kv_heads, context_len, head_dim = keys.shape
    blocks_per_seq = math.ceil(context_len / block_size)
    block_ids = shuffled_block_ids(num_seqs, blocks_per_seq, shuffle, seed)
    keys_host, values_host = np.asarray(keys), np.asarray(values)

    key_cache = np.zeros((num_seqs * blocks_per_seq, num_kv_heads, block_size,
                          head_dim), dtype=keys_host.dtype)
    value_cache = np.zeros_like(key_cache)
    for seq in range(num_seqs):
        for block, block_id in enumerate(block_ids[seq]):
            start = block * block_size
            end = min(start + block_size, context_len)
            key_cache[block_id, :, :end - start] = keys_host[seq, :, start:end]
            value_cache[block_id, :, :end - start] = values_host[seq, :,
                                                                 start:end]

    return (jnp.asarray(key_cache), jnp.asarray(value_cache),
            jnp.asarray(block_ids, dtype=jnp.int32),
            jnp.full((num_seqs,), context_len, dtype=jnp.int32))


def random_kv(num_seqs, num_kv_heads, context_len, head_dim,
              dtype=jnp.float32, seed=1234):
    key_seed, value_seed = jax.random.split(jax.random.key(seed))
    shape = (num_seqs, num_kv_heads, context_len, head_dim)
    return (jax.random.normal(key_seed, shape, dtype=dtype),
            jax.random.normal(value_seed, shape, dtype=dtype))


def random_query(num_seqs, num_heads, head_dim, seed, dtype=jnp.float32):
    rng = np.random.RandomState(seed)
    return jnp.asarray(rng.randn(num_seqs, num_heads, head_dim), dtype)


def assert_close(actual, expected, tolerance=2e-3):
    np.testing.assert_allclose(np.asarray(actual, np.float32),
                               np.asarray(expected, np.float32),
                               rtol=tolerance, atol=tolerance)


def jbench_ms(function, iters=30, warmup=5):
    """Milliseconds for each call, with a wait for the device.

    The warmup runs also take the XLA compilation. It takes seconds, and
    without the warmup all of it goes into the first timed iteration.
    """
    for _ in range(warmup):
        result = function()
    jax.block_until_ready(result)
    start = time.perf_counter()
    for _ in range(iters):
        result = function()
    jax.block_until_ready(result)
    return (time.perf_counter() - start) / iters * 1000


def poisoned_past_context(key_cache, value_cache, block_tables, context_len,
                          block_size, poison=999.0):
    """-> copies of the caches with `poison` in every slot at or after
    context_len. A correct kernel gives the same output with them."""
    key_host = np.asarray(key_cache).copy()
    value_host = np.asarray(value_cache).copy()
    for row in np.asarray(block_tables):
        for block, block_id in enumerate(row):
            first_dead = max(context_len - block * block_size, 0)
            key_host[int(block_id), :, first_dead:] = poison
            value_host[int(block_id), :, first_dead:] = poison
    return jnp.asarray(key_host), jnp.asarray(value_host)


def greedy_workload(model, prompts, osl):
    """-> one token list for each prompt: the prompt, then osl greedy tokens
    from `model`. All the steps use one shape, so XLA compiles one time. The
    padding after the current position cannot change its logits, because the
    mask is causal."""
    sequences = []
    for prompt in prompts:
        length = len(prompt) + osl
        tokens = jnp.zeros((1, length), jnp.int32).at[0, :len(prompt)].set(
            jnp.asarray(prompt, jnp.int32))
        for position in range(len(prompt), length):
            logits, _ = model.forward(tokens, logits_index=jnp.asarray([position - 1]))
            tokens = tokens.at[0, position].set(jnp.argmax(logits[0]))
        sequences.append([int(token) for token in tokens[0]])
    return sequences
