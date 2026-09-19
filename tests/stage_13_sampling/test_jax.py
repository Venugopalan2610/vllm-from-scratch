"""Stage 13 (JAX) - a real batched sampler.

The spec is in app/j13_sampler.py. The checks on the distribution are the
important ones: a sampler that is a little wrong still returns tokens that
look correct.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from app.j13_sampler import (
    SamplingParams,
    apply_repetition_penalty,
    apply_top_k,
    apply_top_p,
    sample,
)
from tests.jhelpers import jbench_ms

VOCAB_SIZE = 512


def _logits(num_rows=4, seed=0, scale=2.0):
    return jax.random.normal(jax.random.key(seed), (num_rows, VOCAB_SIZE),
                             jnp.float32) * scale


def _num_kept(masked_logits):
    return int(np.isfinite(np.asarray(masked_logits)).sum())


def test_greedy_at_temperature_zero(jax_device):
    logits = _logits(4)
    tokens = sample(logits, [SamplingParams(temperature=0.0)] * 4)
    np.testing.assert_array_equal(np.asarray(tokens),
                                  np.asarray(logits.argmax(-1)))


def test_temperature_zero_does_not_poison_the_batch(jax_device):
    """Row 0 is greedy, the others are not. One pass, no branch, no NaN."""
    logits = _logits(4)
    params = [SamplingParams(temperature=0.0),
              SamplingParams(temperature=1.0, seed=1),
              SamplingParams(temperature=0.7, seed=2),
              SamplingParams(temperature=0.0)]
    tokens = np.asarray(sample(logits, params))
    assert tokens[0] == int(logits[0].argmax())
    assert tokens[3] == int(logits[3].argmax())
    assert np.all(tokens >= 0) and np.all(tokens < VOCAB_SIZE), (
        "a NaN got into the argmax")


def test_top_k_keeps_exactly_k(jax_device):
    logits = _logits(3, seed=1)
    masked = np.asarray(apply_top_k(logits, jnp.asarray([1, 5, 0], jnp.int32)))
    assert np.isfinite(masked[0]).sum() == 1
    assert np.isfinite(masked[1]).sum() == 5
    assert np.isfinite(masked[2]).sum() == VOCAB_SIZE, (
        "k=0 must turn the filter off")
    # It must keep the CORRECT tokens.
    top_five = np.argsort(-np.asarray(logits[1]))[:5]
    assert set(np.where(np.isfinite(masked[1]))[0]) == set(top_five)


def test_top_p_is_exact(jax_device):
    """The smallest prefix whose cumulative mass gets to p, and no more."""
    logits = jnp.log(jnp.asarray([[0.4, 0.3, 0.2, 0.05, 0.05]]))
    for p, expected in ((0.4, 1), (0.5, 2), (0.7, 2), (0.75, 3), (0.95, 4)):
        kept = _num_kept(apply_top_p(logits, jnp.asarray([p])))
        assert kept == expected, f"p={p}: kept {kept} tokens, expected {expected}"


def test_top_p_always_keeps_one(jax_device):
    """A peaked row with a very small p must still have a token to sample."""
    logits = jnp.log(jnp.asarray([[0.99, 0.005, 0.005]]))
    assert _num_kept(apply_top_p(logits, jnp.asarray([0.1]))) == 1


def test_repetition_penalty_signs(jax_device):
    """Divide a positive logit, MULTIPLY a negative one. Both must go DOWN."""
    logits = jnp.asarray([[2.0, -2.0, 0.5, -0.5]])
    penalized = np.asarray(apply_repetition_penalty(logits, [[0, 1]],
                                                    jnp.asarray([2.0])))
    assert penalized[0, 0] == pytest.approx(1.0), (
        "divide a positive logit")
    assert penalized[0, 1] == pytest.approx(-4.0), (
        "MULTIPLY a negative logit. A division makes it larger, and that "
        "rewards the token that you wanted to punish.")
    np.testing.assert_allclose(penalized[0, 2:], np.asarray(logits)[0, 2:])


def test_repetition_penalty_of_one_is_a_no_op(jax_device):
    logits = _logits(2, seed=3)
    penalized = apply_repetition_penalty(logits, [[1, 2], [3]],
                                         jnp.asarray([1.0, 1.0]))
    np.testing.assert_allclose(np.asarray(penalized), np.asarray(logits))


def test_a_seed_is_reproducible_and_batch_independent(jax_device):
    """The property that the key model of JAX gives you, checked.

    The same seed must give the same token when the request is alone. It
    must give the same token in a batch of eight, with any other requests.
    """
    logits = _logits(8, seed=4)
    alone = int(sample(logits[3:4], [SamplingParams(temperature=1.0,
                                                    seed=42)])[0])
    params = [SamplingParams(temperature=1.0, seed=seed) for seed in range(8)]
    params[3] = SamplingParams(temperature=1.0, seed=42)
    batched = int(np.asarray(sample(logits, params))[3])
    assert alone == batched, (
        f"seed 42 gave {alone} alone and {batched} in a batch. The seed of a "
        "request must not depend on the other requests of the batch.")

    # Different seeds on the same logits must give different tokens.
    same_rows = jnp.broadcast_to(logits[0], (16, VOCAB_SIZE))
    tokens = np.asarray(sample(same_rows, [SamplingParams(temperature=1.0,
                                                          seed=seed)
                                           for seed in range(16)]))
    assert len(set(tokens.tolist())) > 1, (
        "16 different seeds on the same logits gave one token. The keys are "
        "not different for each row.")


def test_the_distribution_is_right(jax_device):
    """Gumbel-max must be an exact categorical draw, not an
    approximation."""
    probs = np.array([0.5, 0.25, 0.15, 0.07, 0.03])
    logits = jnp.broadcast_to(jnp.log(jnp.asarray(probs)), (4000, 5))
    tokens = np.asarray(sample(logits, [SamplingParams(temperature=1.0,
                                                       seed=seed)
                                        for seed in range(4000)]))
    frequencies = np.bincount(tokens, minlength=5) / len(tokens)

    print("\n  token  expected  sampled")
    for token in range(5):
        print(f"  {token:>5} {probs[token]:>9.3f} {frequencies[token]:>8.3f}")
    np.testing.assert_allclose(frequencies, probs, atol=0.02)


def test_sampler_is_fast_at_batch_64(jax_device):
    """When the kernels are fast, a slow sampler is the bottleneck."""
    logits = _logits(64, seed=5)
    params = [SamplingParams(temperature=0.8, top_k=50, top_p=0.95, seed=seed)
              for seed in range(64)]
    step_ms = jbench_ms(lambda: sample(logits, params), iters=20, warmup=1)
    print(f"\n  sampler at batch 64: {step_ms:.2f} ms/step")
    assert step_ms < 25.0, (
        f"{step_ms:.1f} ms/step is more than the cost of a decode step. Do "
        "you loop over the rows in Python, and not vectorize?")
