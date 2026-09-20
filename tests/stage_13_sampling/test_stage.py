"""Stage 13 - a real batched sampler.

The spec is in app/s13_sampler.py.
"""

import pytest
import torch

from app.s13_sampler import (
    SamplingParams,
    apply_min_p,
    apply_repetition_penalty,
    apply_top_p,
    check_logits,
    sample,
)
from tests.helpers import bench_ms

MASKED = -1e30


def _count_top_choices(logits, params, num_draws):
    top_token = int(logits.argmax())
    return sum(int(sample(logits, [params])[0]) == top_token
               for _ in range(num_draws))


def _num_kept(masked_logits):
    return int((masked_logits[0] > MASKED).sum())


def test_greedy_is_argmax(device):
    logits = torch.randn(8, 100, device=device)
    torch.testing.assert_close(
        sample(logits, [SamplingParams(temperature=0)] * 8), logits.argmax(-1))


def test_temperature_changes_sharpness(device):
    """A low temperature must concentrate on the top token. A high one must
    spread."""
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0]], device=device)
    cold = _count_top_choices(logits, SamplingParams(temperature=0.1), 200)
    hot = _count_top_choices(logits, SamplingParams(temperature=5.0), 200)
    print(f"\n  the top token was selected: temp=0.1 -> {cold}/200,  "
          f"temp=5.0 -> {hot}/200")
    assert cold > hot + 30


@pytest.mark.parametrize("k", [1, 3, 5])
def test_top_k_restricts_the_support(device, k):
    torch.manual_seed(0)
    logits = torch.randn(1, 50, device=device)
    allowed = set(logits[0].topk(k).indices.tolist())
    params = SamplingParams(temperature=1.0, top_k=k)
    drawn = {int(sample(logits, [params])[0]) for _ in range(300)}
    assert drawn <= allowed, f"sampled outside the top-{k}: {drawn - allowed}"


@pytest.mark.parametrize("p,expected", [(0.9, 3), (0.75, 2), (0.4, 1), (1.0, 4)])
def test_top_p_keeps_exactly_the_nucleus(device, p, expected):
    """probs = [0.5, 0.3, 0.15, 0.05]. cumulative = [0.5, 0.8, 0.95, 1.0]."""
    logits = torch.tensor([[0.5, 0.3, 0.15, 0.05]], device=device).log()
    kept = _num_kept(apply_top_p(logits.clone(), torch.tensor([p],
                                                              device=device)))
    assert kept == expected, (
        f"top_p={p} kept {kept} tokens, expected {expected} (the smallest set "
        "whose cumulative probability gets to p)")


def test_top_p_always_keeps_at_least_one(device):
    """A peaked distribution with a very small p must not leave an empty
    nucleus."""
    logits = torch.tensor([[0.97, 0.02, 0.01]], device=device).log()
    assert _num_kept(apply_top_p(logits.clone(),
                                 torch.tensor([0.001], device=device))) >= 1


def test_repetition_penalty_sign_handling(device):
    """Divide a positive logit, multiply a negative logit."""
    logits = torch.tensor([[2.0, -2.0, 0.5]], device=device)
    penalized = apply_repetition_penalty(logits.clone(), [[0, 1]],
                                         torch.tensor([2.0], device=device))
    assert penalized[0, 0].item() == pytest.approx(1.0)    # 2.0 / 2
    assert penalized[0, 1].item() == pytest.approx(-4.0)   # -2.0 * 2
    assert penalized[0, 2].item() == pytest.approx(0.5)    # no change
    print("\n  \033[2mA division of the negative logit gives -1.0. That makes")
    print("  the token MORE probable, not less.\033[0m")


def test_per_request_params_in_one_batch(device):
    """The point: one call, four different configurations."""
    torch.manual_seed(0)
    logits = torch.randn(4, 100, device=device)
    params = [
        SamplingParams(temperature=0),                       # greedy
        SamplingParams(temperature=1.0, top_k=1),            # also greedy
        SamplingParams(temperature=1.0, top_k=5, seed=1),
        SamplingParams(temperature=1.0, top_p=0.9, seed=2),
    ]
    tokens = sample(logits, params)
    assert tokens.shape == (4,)
    assert int(tokens[0]) == int(logits[0].argmax())
    assert int(tokens[1]) == int(logits[1].argmax()), (
        "top_k=1 must be the same as argmax")


def test_seeded_sampling_is_reproducible(device):
    generator = torch.Generator(device=device).manual_seed(5)
    logits = torch.randn(3, 100, device=device, generator=generator)
    params = [SamplingParams(seed=42), SamplingParams(seed=42),
              SamplingParams(seed=7)]
    assert sample(logits, params).tolist() == sample(logits, params).tolist(), (
        "the same seeds must give the same tokens")


def test_same_seed_same_logits_same_token(device):
    """Two equal rows with the same seed must agree.

    (Two rows with the same seed and DIFFERENT logits can differ: the same
    noise, a different distribution.)
    """
    row = torch.randn(1, 100, device=device)
    tokens = sample(torch.cat([row, row], dim=0),
                    [SamplingParams(seed=13), SamplingParams(seed=13)])
    assert int(tokens[0]) == int(tokens[1])


def test_distribution_is_approximately_correct(device):
    """Sampling with no constraint must follow the softmax.

    Gumbel-max is exact, so this must be easily inside the noise.
    """
    probs = torch.tensor([0.6, 0.25, 0.1, 0.05], device=device)
    logits = probs.log().unsqueeze(0).repeat(4000, 1)
    tokens = sample(logits, [SamplingParams(temperature=1.0)] * 4000)
    frequencies = torch.bincount(tokens, minlength=4).float() / 4000

    print(f"\n  target:  {[f'{p:.3f}' for p in probs.tolist()]}")
    print(f"  sampled: {[f'{p:.3f}' for p in frequencies.tolist()]}")
    torch.testing.assert_close(frequencies, probs, rtol=0, atol=0.03)


def test_sampler_is_vectorized_not_looped(device):
    """At batch 256, a Python loop over the requests is slow."""
    torch.manual_seed(0)
    logits = torch.randn(256, 32000, device=device)
    params = [SamplingParams(temperature=1.0, top_k=50, top_p=0.95)
              for _ in range(256)]
    step_ms = bench_ms(lambda: sample(logits, params), iters=20, warmup=5)
    print(f"\n  batch 256, vocab 32000: {step_ms:.2f} ms/step")
    assert step_ms < 25.0, (
        f"{step_ms:.1f} ms for each sampling step is too slow. That is about "
        "a whole decode step. Sort one time for the batch, not one time for "
        "each request.")


def test_check_logits_guards_against_nan(device):
    """NaN logits must be caught immediately with a clear error message.
    A bad quantization scale, a bad FP8 calibration or an overflow in
    attention gives NaN logits. NaN is silent: only a check makes it loud."""
    logits = torch.randn(2, 50, device=device)
    logits[0, 5] = float("nan")
    with pytest.raises(ValueError, match="NaN detected in logits"):
        check_logits(logits)
    with pytest.raises(ValueError, match="NaN detected in logits"):
        sample(logits, [SamplingParams(), SamplingParams()])


def test_min_p_filters_low_probability_relative_to_max(device):
    """min_p scales the cutoff threshold relative to the TOP token's probability.
    probs = [0.8, 0.15, 0.04, 0.01].
    max_prob = 0.8.
    min_p = 0.1 -> cutoff is 0.8 * 0.1 = 0.08.
    Tokens with prob < 0.08 are masked (0.04 and 0.01).
    Tokens kept: 0.8 and 0.15 (2 tokens)."""
    probs = torch.tensor([[0.8, 0.15, 0.04, 0.01]], device=device)
    logits = probs.log()
    kept = _num_kept(apply_min_p(logits.clone(), torch.tensor([0.1], device=device)))
    assert kept == 2
    kept_005 = _num_kept(apply_min_p(logits.clone(), torch.tensor([0.05], device=device)))
    assert kept_005 == 3
    kept_1 = _num_kept(apply_min_p(logits.clone(), torch.tensor([1.0], device=device)))
    assert kept_1 == 1
