"""Stage 13 - A real batched sampler.

Spec in app/s13_sampler.py.
"""

import pytest
import torch

from app.s13_sampler import (
    SamplingParams,
    apply_repetition_penalty,
    apply_top_k,
    apply_top_p,
    sample,
)
from tests.helpers import bench_ms

SP = SamplingParams


def test_greedy_is_argmax(dev):
    lg = torch.randn(8, 100, device=dev)
    got = sample(lg, [SP(temperature=0)] * 8)
    torch.testing.assert_close(got, lg.argmax(-1))


def test_temperature_changes_sharpness(dev):
    """Low temperature should concentrate on the top token, high should spread."""
    lg = torch.tensor([[3.0, 2.0, 1.0, 0.0]], device=dev)
    top = int(lg.argmax())
    cold = sum(int(sample(lg, [SP(temperature=0.1)])[0]) == top for _ in range(200))
    hot = sum(int(sample(lg, [SP(temperature=5.0)])[0]) == top for _ in range(200))
    print(f"\n  top token chosen: temp=0.1 -> {cold}/200,  temp=5.0 -> {hot}/200")
    assert cold > hot + 30


@pytest.mark.parametrize("k", [1, 3, 5])
def test_top_k_restricts_the_support(dev, k):
    torch.manual_seed(0)
    lg = torch.randn(1, 50, device=dev)
    allowed = set(lg[0].topk(k).indices.tolist())
    draws = {int(sample(lg, [SP(temperature=1.0, top_k=k)])[0]) for _ in range(300)}
    assert draws <= allowed, f"sampled outside the top-{k}: {draws - allowed}"


@pytest.mark.parametrize("p,expected", [(0.9, 3), (0.75, 2), (0.4, 1), (1.0, 4)])
def test_top_p_keeps_exactly_the_nucleus(dev, p, expected):
    """probs = [0.5, 0.3, 0.15, 0.05]; cumulative = [0.5, 0.8, 0.95, 1.0]."""
    logits = torch.tensor([[0.5, 0.3, 0.15, 0.05]], device=dev).log()
    out = apply_top_p(logits.clone(), torch.tensor([p], device=dev))
    kept = int((out[0] > -1e30).sum())
    assert kept == expected, (
        f"top_p={p} kept {kept} tokens, expected {expected} "
        "(smallest set whose cumulative probability reaches p)"
    )


def test_top_p_always_keeps_at_least_one(dev):
    """A peaked distribution with a tiny p must not leave an empty nucleus."""
    logits = torch.tensor([[0.97, 0.02, 0.01]], device=dev).log()
    out = apply_top_p(logits.clone(), torch.tensor([0.001], device=dev))
    assert int((out[0] > -1e30).sum()) >= 1


def test_repetition_penalty_sign_handling(dev):
    """Positive logits divide, negative logits multiply."""
    logits = torch.tensor([[2.0, -2.0, 0.5]], device=dev)
    pen = torch.tensor([2.0], device=dev)
    out = apply_repetition_penalty(logits.clone(), [[0, 1]], pen)
    assert out[0, 0].item() == pytest.approx(1.0)    # 2.0 / 2
    assert out[0, 1].item() == pytest.approx(-4.0)   # -2.0 * 2
    assert out[0, 2].item() == pytest.approx(0.5)    # untouched
    print("\n  \033[2mIf you divided the negative logit you would get -1.0, i.e.")
    print("  you would have PROMOTED the token you meant to suppress.\033[0m")


def test_per_request_params_in_one_batch(dev):
    """The whole point: one call, four different configurations."""
    torch.manual_seed(0)
    lg = torch.randn(4, 100, device=dev)
    params = [
        SP(temperature=0),                       # greedy
        SP(temperature=1.0, top_k=1),            # also effectively greedy
        SP(temperature=1.0, top_k=5, seed=1),
        SP(temperature=1.0, top_p=0.9, seed=2),
    ]
    got = sample(lg, params)
    assert got.shape == (4,)
    assert int(got[0]) == int(lg[0].argmax())
    assert int(got[1]) == int(lg[1].argmax()), "top_k=1 must collapse to argmax"


def test_seeded_sampling_is_reproducible(dev):
    lg = torch.randn(3, 100, device=dev,
                     generator=torch.Generator(device=dev).manual_seed(5))
    a = sample(lg, [SP(seed=42), SP(seed=42), SP(seed=7)])
    b = sample(lg, [SP(seed=42), SP(seed=42), SP(seed=7)])
    assert a.tolist() == b.tolist(), "same seeds must give the same tokens"


def test_same_seed_same_logits_same_token(dev):
    """Two identical rows with the same seed must agree.

    (Two rows with the same seed but DIFFERENT logits need not agree -- same
    noise, different distribution.)
    """
    row = torch.randn(1, 100, device=dev)
    lg = torch.cat([row, row], dim=0)
    got = sample(lg, [SP(seed=13), SP(seed=13)])
    assert int(got[0]) == int(got[1])


def test_distribution_is_approximately_correct(dev):
    """Unconstrained sampling must actually follow the softmax.

    Gumbel-max is exact, so this should be comfortably within noise.
    """
    probs = torch.tensor([0.6, 0.25, 0.1, 0.05], device=dev)
    logits = probs.log().unsqueeze(0).repeat(4000, 1)
    got = sample(logits, [SP(temperature=1.0)] * 4000)
    counts = torch.bincount(got, minlength=4).float() / 4000

    print(f"\n  target:  {[f'{p:.3f}' for p in probs.tolist()]}")
    print(f"  sampled: {[f'{p:.3f}' for p in counts.tolist()]}")
    torch.testing.assert_close(counts, probs, rtol=0, atol=0.03)


def test_sampler_is_vectorized_not_looped(dev):
    """At batch 256 a per-request Python loop is measurably slow."""
    torch.manual_seed(0)
    lg = torch.randn(256, 32000, device=dev)
    params = [SP(temperature=1.0, top_k=50, top_p=0.95) for _ in range(256)]
    ms = bench_ms(lambda: sample(lg, params), iters=20, warmup=5)
    print(f"\n  batch 256, vocab 32000: {ms:.2f} ms/step")
    assert ms < 25.0, (
        f"{ms:.1f} ms per sampling step is too slow -- that is on the order of "
        "a whole decode step. Sort once for the batch, not once per request."
    )
