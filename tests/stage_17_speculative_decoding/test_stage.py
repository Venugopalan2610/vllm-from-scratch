"""Stage 17 - speculative decoding.

The spec is in app/s17_speculative.py. The main check is statistical. It
proves that the output distribution does not move. The rest is bookkeeping.
"""

import random

import pytest
import torch

from app.s17_speculative import (
    SpeculativeStats,
    expected_speedup,
    ngram_propose,
    rejection_sample,
)


def inverse_cdf_draw(probs, rng):
    threshold, cumulative = rng.random(), 0.0
    for token, prob in enumerate(probs.tolist()):
        cumulative += prob
        if threshold < cumulative:
            return token
    return len(probs) - 1


def uniform_rows(num_rows, vocab_size):
    return torch.full((num_rows, vocab_size), 1.0 / vocab_size)


# ---- the n-gram proposal --------------------------------------------

def test_ngram_copies_after_the_most_recent_match():
    assert ngram_propose([1, 2, 3, 4, 1, 2, 3], max_draft=3, ngram=3) == [4, 1, 2]


def test_ngram_returns_empty_without_a_match():
    assert ngram_propose([5, 6, 7, 8], max_draft=3, ngram=3) == []
    assert ngram_propose([1, 2], max_draft=3, ngram=3) == []


def test_ngram_prefers_the_most_recent_occurrence():
    draft = ngram_propose([7, 7, 7, 0, 0, 0, 7, 7, 7], max_draft=1, ngram=3)
    assert draft == [0], (
        f"expected the most recent [7,7,7] -> 0, got {draft}")


# ---- rejection sampling ---------------------------------------------

def test_identical_distributions_accept_everything():
    """If the draft IS the target, nothing is ever rejected."""
    rng = random.Random(0)
    vocab_size, num_drafts = 6, 4
    target_probs = uniform_rows(num_drafts + 1, vocab_size)
    draft_probs = uniform_rows(num_drafts, vocab_size)
    for _ in range(200):
        draft = [rng.randrange(vocab_size) for _ in range(num_drafts)]
        emitted, num_accepted = rejection_sample(target_probs, draft_probs,
                                                 draft, rng)
        assert num_accepted == num_drafts
        assert emitted[:num_drafts] == draft
        assert len(emitted) == num_drafts + 1, (
            "when all are accepted, it must also emit the bonus token")


def test_rejection_stops_at_the_first_failure():
    """A rejection removes a token. Everything after it depended on that
    token."""
    rng = random.Random(0)
    # The draft is certain about token 0. The target is certain about token 1.
    target_probs = torch.tensor([[0.0, 1.0], [0.5, 0.5], [0.5, 0.5]])
    draft_probs = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    emitted, num_accepted = rejection_sample(target_probs, draft_probs, [0, 0],
                                             rng)
    assert num_accepted == 0, (
        "a token with a target probability of 0 must always be rejected")
    assert emitted == [1], "emit only the correction, then stop"


def test_output_distribution_is_exactly_the_target():
    """The whole reason for the method.

    The draft here is very bad on purpose: it prefers exactly the tokens that
    the target does not. The emitted distribution must still be the target.
    """
    target_probs = torch.tensor([[0.6, 0.2, 0.15, 0.05],
                                 [0.25, 0.25, 0.25, 0.25]])
    draft_probs = torch.tensor([[0.1, 0.4, 0.3, 0.2]])
    rng = random.Random(0)
    num_trials = 120000
    counts = [0] * 4
    for _ in range(num_trials):
        draft_token = inverse_cdf_draw(draft_probs[0], rng)
        emitted, _ = rejection_sample(target_probs, draft_probs, [draft_token],
                                      rng)
        counts[emitted[0]] += 1

    frequencies = torch.tensor([count / num_trials for count in counts])
    print(f"\n  draft q:   {[f'{x:.3f}' for x in draft_probs[0].tolist()]}")
    print(f"  target p:  {[f'{x:.3f}' for x in target_probs[0].tolist()]}")
    print(f"  emitted:   {[f'{x:.3f}' for x in frequencies.tolist()]}")
    torch.testing.assert_close(frequencies, target_probs[0], rtol=0, atol=0.006)
    print("\n  \033[1mLossless: a bad draft costs speed, never quality.\033[0m")
    print("  \033[2mRemove the residual term, and this check fails while every")
    print("  other check in the file still passes.\033[0m")


def test_acceptance_rate_tracks_draft_quality():
    """The verifier accepts a good draft frequently, and a bad draft
    rarely."""
    rng = random.Random(0)
    vocab_size, num_drafts = 4, 3

    def acceptance_rate(draft_row, num_trials=3000):
        target_probs = uniform_rows(num_drafts + 1, vocab_size)
        draft_probs = draft_row.unsqueeze(0).repeat(num_drafts, 1)
        stats = SpeculativeStats()
        for _ in range(num_trials):
            draft = [inverse_cdf_draw(draft_row, rng)
                     for _ in range(num_drafts)]
            _, num_accepted = rejection_sample(target_probs, draft_probs, draft,
                                               rng)
            stats.record(num_drafts, num_accepted)
        return stats.acceptance_rate

    matched = acceptance_rate(torch.full((vocab_size,), 1.0 / vocab_size))
    mismatched = acceptance_rate(torch.tensor([0.97, 0.01, 0.01, 0.01]))
    print(f"\n  matched draft:     acceptance {matched:.2f}")
    print(f"  mismatched draft:  acceptance {mismatched:.2f}")
    assert matched > 0.95
    assert mismatched < matched


def test_stats_bookkeeping():
    stats = SpeculativeStats()
    stats.record(4, 4)
    stats.record(4, 0)
    assert stats.proposed == 8
    assert stats.accepted == 4
    assert stats.acceptance_rate == pytest.approx(0.5)
    # 4+1 tokens, then 0+1 tokens, over 2 target calls
    assert stats.tokens_per_target_call == pytest.approx(3.0)


def test_expected_speedup_shape():
    assert expected_speedup(0.0, 4) == pytest.approx(1.0), (
        "no acceptance must give ordinary decoding, not worse")
    assert expected_speedup(1.0, 4) == pytest.approx(5.0), (
        "perfect acceptance gives k+1 tokens for each target call")

    rows = [(num_draft, expected_speedup(0.7, num_draft))
            for num_draft in (1, 2, 4, 8, 16)]
    print("\n  acceptance rate 0.7:")
    for num_draft, tokens in rows:
        print(f"    k={num_draft:>2}  ->  {tokens:.2f} tokens for each target "
              "call")
    gain_from_4_to_8 = rows[3][1] - rows[2][1]
    gain_from_1_to_2 = rows[1][1] - rows[0][1]
    assert gain_from_4_to_8 < gain_from_1_to_2, (
        "expected a smaller gain as k increases")
    print("\n  \033[2mThe gain decreases: the 8th token needs 8 acceptances in")
    print("  sequence. After about k=4, you mostly pay the draft cost for")
    print("  tokens that you discard.\033[0m")


def test_draft_cost_is_charged():
    free = expected_speedup(0.8, 4, draft_cost_ratio=0.0)
    costly = expected_speedup(0.8, 4, draft_cost_ratio=0.2)
    assert costly < free, "the draft model is not free"
