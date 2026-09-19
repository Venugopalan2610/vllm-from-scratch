"""Stage 12 (JAX) - no more compilation, with shape buckets.

The spec:

    app/j12_compiled.py must define

        CompiledDecode(fn, buckets=(1, 2, 4, 8), donate=())
            .compile(make_inputs) / .run(*inputs) / .run_eager(*inputs)
            .num_compiled / .bucket_for(bs) / .replays / .eager_calls

The main check: a hundred calls at changing batch sizes cause exactly zero
compilations after startup.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from app.j12_compiled import CompiledDecode
from tests.jhelpers import jbench_ms
from tests.helpers import elapsed_ms

BUCKETS = (1, 2, 4, 8)
WIDTH = 512


def _step(inputs, weight):
    """A stand-in for a decode step: batch in, batch out, one weight read."""
    return jnp.tanh(inputs @ weight) @ weight.T


def _weights(seed=0):
    return jax.random.normal(jax.random.key(seed), (WIDTH, WIDTH),
                             jnp.float32) * 0.02


def _zeros(batch_size):
    return jnp.zeros((batch_size, WIDTH), jnp.float32)


def _compiled(function, weight, buckets=BUCKETS, donate=()):
    compiled = CompiledDecode(function, buckets=buckets, donate=donate)
    compiled.compile(lambda batch_size: (_zeros(batch_size), weight))
    return compiled


def test_bucket_for(jax_device):
    compiled = CompiledDecode(_step, buckets=BUCKETS)
    assert [compiled.bucket_for(size) for size in (1, 2, 3, 4, 5, 8)] == [
        1, 2, 4, 4, 8, 8]
    with pytest.raises(ValueError):
        compiled.bucket_for(9)


def test_one_executable_per_bucket(jax_device):
    compiled = _compiled(_step, _weights())
    assert compiled.num_compiled == len(BUCKETS), (
        f"{compiled.num_compiled} executables for {len(BUCKETS)} buckets")


def test_padding_does_not_change_the_answer(jax_device):
    """A batch of 3 runs in the bucket-4 program and still returns 3 rows."""
    weight = _weights()
    compiled = _compiled(_step, weight)
    for batch_size in (1, 2, 3, 5, 7, 8):
        inputs = jax.random.normal(jax.random.key(batch_size),
                                   (batch_size, WIDTH), jnp.float32)
        output = compiled.run(inputs, weight)
        assert output.shape == (batch_size, WIDTH), (
            f"batch {batch_size} came back as {output.shape}")
        np.testing.assert_allclose(np.asarray(output),
                                   np.asarray(_step(inputs, weight)),
                                   rtol=1e-4, atol=1e-4)


def test_no_recompiles_across_wandering_batch_sizes(jax_device):
    """The point of the stage.

    A hundred calls at every batch size from 1 to 8. After startup the
    compile count must not change one time.
    """
    weight = _weights()
    compiled = _compiled(_step, weight)
    rng = np.random.RandomState(0)
    for _ in range(100):
        compiled.run(_zeros(int(rng.randint(1, 9))), weight)
    jax.block_until_ready(weight)

    # The cost of the path with no buckets.
    unbucketed = jax.jit(_step)
    compiles_before = unbucketed._cache_size()
    for batch_size in range(1, 9):
        unbucketed(_zeros(batch_size), weight)
    unbucketed_compiles = unbucketed._cache_size() - compiles_before

    print(f"\n  100 calls with buckets, batch 1-8:  {compiled.replays} runs, "
          f"{compiled.num_compiled} executables, 0 new compiles")
    print(f"  the same 8 shapes with no buckets:  {unbucketed_compiles} "
          "compilations")
    assert compiled.replays == 100
    assert compiled.num_compiled == len(BUCKETS), (
        f"num_compiled went from {len(BUCKETS)} to {compiled.num_compiled}. "
        "Something in run() compiles on each call. Do you call jax.jit inside "
        "run(), and not use the executable that you already made?")
    assert unbucketed_compiles >= 8, (
        "expected one compilation for each shape with no buckets")


def test_one_recompile_costs_more_than_many_steps(jax_device):
    """The exchange rate that this stage fixes.

    Not "buckets are faster than no buckets": when warm, both run the same
    program. The point is the cost of a MISS. Measure one compilation
    against one warm step. The answer is the number of requests that you
    delay each time a new batch size comes.
    """
    weight = _weights()

    # A new computation, so that nothing in this session compiled it before
    # and made the measurement a cache hit.
    def deep_step(inputs, weight):
        for _ in range(8):
            inputs = jnp.tanh(inputs @ weight) @ weight.T
        return inputs

    compiled, total_compile_ms = elapsed_ms(
        lambda: _compiled(deep_step, weight))
    compile_ms = total_compile_ms / len(BUCKETS)
    step_ms = jbench_ms(lambda: compiled.run(_zeros(4), weight), iters=50)

    print(f"\n  one compilation: {compile_ms:8.1f} ms")
    print(f"  one warm step:   {step_ms:8.3f} ms")
    print(f"\n  \033[1mOne batch size with no bucket costs "
          f"{compile_ms / step_ms:.0f} steps of GPU time.\033[0m")
    print("  \033[2mAnd it is paid on the critical path, by the request that")
    print("  arrived at a batch size that you did not see before.\033[0m")
    assert compile_ms > step_ms * 20, (
        f"a compile ({compile_ms:.1f} ms) must be much larger than a step "
        f"({step_ms:.3f} ms). If not, .compile() probably does not compile. "
        "Does it wait for the first run()?")


def test_donation_reuses_the_buffer(jax_device):
    """You can see the donation. JAX makes the donated array invalid.

    That proves that XLA used the buffer again in place, and did not copy
    it. It is
    also the trap: a read of a donated buffer after the call raises an error.
    """
    weight = _weights()

    def step_in_place(state, weight):
        return state + jnp.tanh(state @ weight) @ weight.T

    compiled = _compiled(step_in_place, weight, buckets=(4,), donate=(0,))
    state = jnp.ones((4, WIDTH), jnp.float32)
    assert compiled.run(state, weight).shape == (4, WIDTH)

    with pytest.raises(Exception) as error:
        np.asarray(state)
    print(f"\n  a read of the donated buffer after the call raises: "
          f"{type(error.value).__name__}")
    print("  \033[2mThat is the aliasing. For a KV cache it is the difference")
    print("  between a rewrite of the whole buffer at every step and an")
    print("  update where the buffer already is.\033[0m")


def test_eager_fallback(jax_device):
    """A batch larger than every bucket must still be possible to serve."""
    weight = _weights()
    compiled = _compiled(_step, weight)
    inputs = jax.random.normal(jax.random.key(9), (16, WIDTH), jnp.float32)
    with pytest.raises(ValueError):
        compiled.run(inputs, weight)
    np.testing.assert_allclose(np.asarray(compiled.run_eager(inputs, weight)),
                               np.asarray(_step(inputs, weight)),
                               rtol=1e-4, atol=1e-4)
    assert compiled.eager_calls == 1
