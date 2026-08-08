"""Stage 13 (JAX) - A real batched sampler.

Spec in app/j13_sampler.py. The distributional checks are the ones that matter:
a sampler that is subtly wrong still returns plausible tokens.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from app.j13_sampler import (SamplingParams, apply_repetition_penalty,
                             apply_top_k, apply_top_p, row_keys, sample)

V = 512


def _logits(B=4, seed=0, scale=2.0):
    return jax.random.normal(jax.random.key(seed), (B, V), jnp.float32) * scale


def test_greedy_at_temperature_zero(jdev):
    lg = _logits(4)
    params = [SamplingParams(temperature=0.0) for _ in range(4)]
    got = sample(lg, params)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(lg.argmax(-1)))


def test_temperature_zero_does_not_poison_the_batch(jdev):
    """Row 0 is greedy, the rest are not. One pass, no branching, no nans."""
    lg = _logits(4)
    params = [SamplingParams(temperature=0.0), SamplingParams(temperature=1.0, seed=1),
              SamplingParams(temperature=0.7, seed=2), SamplingParams(temperature=0.0)]
    got = np.asarray(sample(lg, params))
    assert got[0] == int(lg[0].argmax())
    assert got[3] == int(lg[3].argmax())
    assert np.all(got >= 0) and np.all(got < V), "nan leaked into the argmax"


def test_top_k_keeps_exactly_k(jdev):
    lg = _logits(3, seed=1)
    k = jnp.asarray([1, 5, 0], jnp.int32)
    out = np.asarray(apply_top_k(lg, k))
    assert np.isfinite(out[0]).sum() == 1
    assert np.isfinite(out[1]).sum() == 5
    assert np.isfinite(out[2]).sum() == V, "k=0 must disable the filter"
    # and it kept the RIGHT ones
    top5 = np.argsort(-np.asarray(lg[1]))[:5]
    assert set(np.where(np.isfinite(out[1]))[0]) == set(top5)


def test_top_p_is_exact(jdev):
    """The smallest prefix whose cumulative mass reaches p, and no more."""
    lg = jnp.log(jnp.asarray([[0.4, 0.3, 0.2, 0.05, 0.05]]))
    for p, expect in ((0.4, 1), (0.5, 2), (0.7, 2), (0.75, 3), (0.95, 4)):
        out = np.asarray(apply_top_p(lg, jnp.asarray([p])))
        n = int(np.isfinite(out).sum())
        assert n == expect, f"p={p}: kept {n} tokens, expected {expect}"


def test_top_p_always_keeps_one(jdev):
    """A peaked row with a tiny p must still be samplable."""
    lg = jnp.log(jnp.asarray([[0.99, 0.005, 0.005]]))
    out = np.asarray(apply_top_p(lg, jnp.asarray([0.1])))
    assert int(np.isfinite(out).sum()) == 1


def test_repetition_penalty_signs(jdev):
    """Divide positives, MULTIPLY negatives. Both must move DOWN."""
    lg = jnp.asarray([[2.0, -2.0, 0.5, -0.5]])
    out = np.asarray(apply_repetition_penalty(lg, [[0, 1]], jnp.asarray([2.0])))
    assert out[0, 0] == pytest.approx(1.0), "positive logit should be divided"
    assert out[0, 1] == pytest.approx(-4.0), (
        "negative logit should be MULTIPLIED. Dividing it makes it larger, "
        "which rewards the token you meant to punish."
    )
    np.testing.assert_allclose(out[0, 2:], np.asarray(lg)[0, 2:])


def test_repetition_penalty_of_one_is_a_no_op(jdev):
    lg = _logits(2, seed=3)
    out = apply_repetition_penalty(lg, [[1, 2], [3]], jnp.asarray([1.0, 1.0]))
    np.testing.assert_allclose(np.asarray(out), np.asarray(lg))


def test_a_seed_is_reproducible_and_batch_independent(jdev):
    """The property JAX's key model gives you for free, checked.

    The same seed must produce the same token whether the request is alone or
    sitting in a batch of eight, and whoever its neighbours are.
    """
    lg = _logits(8, seed=4)
    alone = int(sample(lg[3:4], [SamplingParams(temperature=1.0, seed=42)])[0])

    params = [SamplingParams(temperature=1.0, seed=i) for i in range(8)]
    params[3] = SamplingParams(temperature=1.0, seed=42)
    batched = int(np.asarray(sample(lg, params))[3])

    assert alone == batched, (
        f"seed 42 gave {alone} alone and {batched} in a batch. A per-request "
        "seed must not depend on who else is in the batch."
    )

    # different seeds on identical logits must actually differ
    same = jnp.broadcast_to(lg[0], (16, V))
    got = np.asarray(sample(same, [SamplingParams(temperature=1.0, seed=i)
                                   for i in range(16)]))
    assert len(set(got.tolist())) > 1, (
        "16 different seeds on identical logits produced one token -- the "
        "per-row keys are not actually per row"
    )


def test_the_distribution_is_right(jdev):
    """Gumbel-max must be an exact categorical draw, not an approximation."""
    probs = np.array([0.5, 0.25, 0.15, 0.07, 0.03])
    lg = jnp.broadcast_to(jnp.log(jnp.asarray(probs)), (4000, 5))
    params = [SamplingParams(temperature=1.0, seed=i) for i in range(4000)]
    got = np.asarray(sample(lg, params))
    freq = np.bincount(got, minlength=5) / len(got)

    print("\n  token   want    got")
    for i in range(5):
        print(f"  {i:>5} {probs[i]:>7.3f} {freq[i]:>6.3f}")
    np.testing.assert_allclose(freq, probs, atol=0.02)


def test_sampler_is_fast_at_batch_64(jdev):
    """Once the kernels are fast, a slow sampler is the bottleneck."""
    lg = _logits(64, seed=5)
    params = [SamplingParams(temperature=0.8, top_k=50, top_p=0.95, seed=i)
              for i in range(64)]

    sample(lg, params)                       # warm
    jax.block_until_ready(lg)
    t = time.perf_counter()
    for _ in range(20):
        out = sample(lg, params)
    jax.block_until_ready(out)
    ms = (time.perf_counter() - t) / 20 * 1000

    print(f"\n  sampler at batch 64: {ms:.2f} ms/step")
    assert ms < 25.0, (
        f"{ms:.1f} ms/step is more than a decode step costs. Are you looping "
        "over rows in Python instead of vectorizing?"
    )
