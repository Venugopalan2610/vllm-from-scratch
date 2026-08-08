"""Stage 12 (JAX) - Killing recompilation with shape buckets.

The spec:

    app/j12_compiled.py must define

        CompiledDecode(fn, buckets=(1, 2, 4, 8), donate=())
            .compile(make_inputs) / .run(*inputs) / .run_eager(*inputs)
            .num_compiled / .bucket_for(bs) / .replays / .eager_calls

The headline check is that a hundred calls at wandering batch sizes cause
exactly zero compilations after startup.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from app.j12_compiled import CompiledDecode

BUCKETS = (1, 2, 4, 8)
H = 512


def _step(x, w):
    """Stands in for a decode step: batch in, batch out, one weight read."""
    return jnp.tanh(x @ w) @ w.T


def _weights(key=0):
    return jax.random.normal(jax.random.key(key), (H, H), jnp.float32) * 0.02


def _make_inputs(w):
    def make(bs):
        return (jnp.zeros((bs, H), jnp.float32), w)
    return make


def test_bucket_for(jdev):
    r = CompiledDecode(_step, buckets=BUCKETS)
    assert [r.bucket_for(n) for n in (1, 2, 3, 4, 5, 8)] == [1, 2, 4, 4, 8, 8]
    with pytest.raises(ValueError):
        r.bucket_for(9)


def test_one_executable_per_bucket(jdev):
    w = _weights()
    r = CompiledDecode(_step, buckets=BUCKETS)
    r.compile(_make_inputs(w))
    assert r.num_compiled == len(BUCKETS), (
        f"{r.num_compiled} executables for {len(BUCKETS)} buckets"
    )


def test_padding_does_not_change_the_answer(jdev):
    """A batch of 3 runs in the bucket-4 program and still returns 3 rows."""
    w = _weights()
    r = CompiledDecode(_step, buckets=BUCKETS)
    r.compile(_make_inputs(w))

    for bs in (1, 2, 3, 5, 7, 8):
        x = jax.random.normal(jax.random.key(bs), (bs, H), jnp.float32)
        got = r.run(x, w)
        want = _step(x, w)
        assert got.shape == (bs, H), f"batch {bs} came back as {got.shape}"
        np.testing.assert_allclose(np.asarray(got), np.asarray(want),
                                   rtol=1e-4, atol=1e-4)


def test_no_recompiles_across_wandering_batch_sizes(jdev):
    """The whole point of the stage.

    A hundred calls at every batch size between 1 and 8. After startup the
    compile count must not move once.
    """
    w = _weights()
    r = CompiledDecode(_step, buckets=BUCKETS)
    r.compile(_make_inputs(w))

    probe = jax.jit(_step)
    before = probe._cache_size()
    rng = np.random.RandomState(0)
    for _ in range(100):
        bs = int(rng.randint(1, 9))
        r.run(jnp.zeros((bs, H), jnp.float32), w)
    jax.block_until_ready(w)

    # and what the un-bucketed path would have cost
    for bs in range(1, 9):
        probe(jnp.zeros((bs, H), jnp.float32), w)
    grew = probe._cache_size() - before

    print(f"\n  100 bucketed calls, batch 1-8:  {r.replays} runs, "
          f"{r.num_compiled} executables, 0 new compiles")
    print(f"  the same 8 shapes, un-bucketed: {grew} compilations")
    assert r.replays == 100
    assert r.num_compiled == len(BUCKETS), (
        f"num_compiled went from {len(BUCKETS)} to {r.num_compiled} -- "
        "something in run() is compiling per call. Are you calling jax.jit "
        "inside run() instead of using the executable you already built?"
    )
    assert grew >= 8, "expected the un-bucketed probe to compile per shape"


def test_one_recompile_costs_more_than_many_steps(jdev):
    """The exchange rate this stage exists to fix.

    Not "bucketed is faster than un-bucketed" -- both run the same program once
    they are warm. The point is what a MISS costs. Measure a single compilation
    against a single steady-state step, and the answer is how many requests you
    delay every time a new batch size shows up.
    """
    w = _weights()

    # a fresh computation, so nothing in this session can already have
    # compiled it and turn the measurement into a cache hit
    def deep_step(x, weight):
        for _ in range(8):
            x = jnp.tanh(x @ weight) @ weight.T
        return x

    r = CompiledDecode(deep_step, buckets=BUCKETS)
    t = time.perf_counter()
    r.compile(lambda bs: (jnp.zeros((bs, H), jnp.float32), w))
    compile_ms = (time.perf_counter() - t) * 1000 / len(BUCKETS)

    x = jnp.zeros((4, H), jnp.float32)
    for _ in range(5):
        out = r.run(x, w)
    jax.block_until_ready(out)
    t = time.perf_counter()
    for _ in range(50):
        out = r.run(x, w)
    jax.block_until_ready(out)
    step_ms = (time.perf_counter() - t) * 1000 / 50

    print(f"\n  one compilation: {compile_ms:8.1f} ms")
    print(f"  one warm step:   {step_ms:8.3f} ms")
    print(f"\n  \033[1mA single unbucketed batch size costs "
          f"{compile_ms / step_ms:.0f} steps' worth of GPU time.\033[0m")
    print("  \033[2mAnd it is paid on the critical path, by whichever request")
    print("  happened to arrive at a batch size you had not seen.\033[0m")
    assert compile_ms > step_ms * 20, (
        f"a compile ({compile_ms:.1f} ms) should dwarf a step "
        f"({step_ms:.3f} ms); if it does not, .compile() is probably not "
        "actually compiling -- is it deferring to the first run() instead?"
    )


def test_donation_reuses_the_buffer(jdev):
    """Donation is visible: the donated array is invalidated afterwards.

    That is the proof it was reused in place rather than copied. It is also
    the trap -- read a donated buffer after the call and JAX raises.
    """
    w = _weights()

    def step_inplace(state, weight):
        return state + jnp.tanh(state @ weight) @ weight.T

    r = CompiledDecode(step_inplace, buckets=(4,), donate=(0,))
    r.compile(lambda bs: (jnp.zeros((bs, H), jnp.float32), w))

    state = jnp.ones((4, H), jnp.float32)
    out = r.run(state, w)
    assert out.shape == (4, H)

    with pytest.raises(Exception) as e:
        _ = np.asarray(state)
    print(f"\n  reading the donated buffer afterwards raises: "
          f"{type(e.value).__name__}")
    print("  \033[2mThat is the aliasing working. For a KV cache it is the")
    print("  difference between rewriting the whole buffer every step and")
    print("  updating it where it already lives.\033[0m")


def test_eager_fallback(jdev):
    """A batch bigger than every bucket still has to be servable."""
    w = _weights()
    r = CompiledDecode(_step, buckets=BUCKETS)
    r.compile(_make_inputs(w))

    x = jax.random.normal(jax.random.key(9), (16, H), jnp.float32)
    with pytest.raises(ValueError):
        r.run(x, w)
    got = r.run_eager(x, w)
    np.testing.assert_allclose(np.asarray(got), np.asarray(_step(x, w)),
                               rtol=1e-4, atol=1e-4)
    assert r.eager_calls == 1
