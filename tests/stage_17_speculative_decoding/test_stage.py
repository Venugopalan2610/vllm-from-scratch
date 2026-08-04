"""Stage 17 - Speculative decoding.

Spec in app/s17_speculative.py. The headline test is statistical: prove the
output distribution is unchanged. Everything else is bookkeeping.
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


def draw(probs, rng):
    r, c = rng.random(), 0.0
    for i, p in enumerate(probs.tolist()):
        c += p
        if r < c:
            return i
    return len(probs) - 1


# ---- n-gram proposal ------------------------------------------------

def test_ngram_copies_after_the_most_recent_match():
    assert ngram_propose([1, 2, 3, 4, 1, 2, 3], k=3, n=3) == [4, 1, 2]


def test_ngram_returns_empty_without_a_match():
    assert ngram_propose([5, 6, 7, 8], k=3, n=3) == []
    assert ngram_propose([1, 2], k=3, n=3) == []


def test_ngram_prefers_the_most_recent_occurrence():
    toks = [7, 7, 7, 0, 0, 0, 7, 7, 7]
    got = ngram_propose(toks, k=1, n=3)
    assert got == [0], f"expected the most recent [7,7,7] -> 0, got {got}"


# ---- rejection sampling ---------------------------------------------

def test_identical_distributions_accept_everything():
    """If the draft IS the target, nothing is ever rejected."""
    rng = random.Random(0)
    V, K = 6, 4
    p = torch.full((K + 1, V), 1.0 / V)
    q = torch.full((K, V), 1.0 / V)
    for _ in range(200):
        toks = [rng.randrange(V) for _ in range(K)]
        out, n = rejection_sample(p, q, toks, rng)
        assert n == K
        assert out[:K] == toks
        assert len(out) == K + 1, "all-accepted must also emit the bonus token"


def test_rejection_stops_at_the_first_failure():
    """Everything after a rejection was conditioned on a dead token."""
    rng = random.Random(0)
    # draft is certain about token 0; target is certain about token 1
    p = torch.tensor([[0.0, 1.0], [0.5, 0.5], [0.5, 0.5]])
    q = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    out, n = rejection_sample(p, q, [0, 0], rng)
    assert n == 0, "a token with target probability 0 must always be rejected"
    assert len(out) == 1, "emit only the correction, then stop"
    assert out[0] == 1


def test_output_distribution_is_exactly_the_target():
    """The whole justification for the technique.

    The draft here is deliberately terrible -- it favours exactly the tokens
    the target does not. The emitted distribution must still be the target's.
    """
    p = torch.tensor([[0.6, 0.2, 0.15, 0.05], [0.25, 0.25, 0.25, 0.25]])
    q = torch.tensor([[0.1, 0.4, 0.3, 0.2]])
    rng = random.Random(0)
    N = 120000
    counts = [0] * 4
    for _ in range(N):
        dt = draw(q[0], rng)
        out, _ = rejection_sample(p, q, [dt], rng)
        counts[out[0]] += 1

    emp = torch.tensor([c / N for c in counts])
    print(f"\n  draft q:   {[f'{x:.3f}' for x in q[0].tolist()]}")
    print(f"  target p:  {[f'{x:.3f}' for x in p[0].tolist()]}")
    print(f"  emitted:   {[f'{x:.3f}' for x in emp.tolist()]}")
    torch.testing.assert_close(emp, p[0], rtol=0, atol=0.006)
    print("\n  \033[1mLossless: a bad draft costs speed, never quality.\033[0m")
    print("  \033[2mDrop the residual term and this test fails while every")
    print("  other test in the file still passes.\033[0m")


def test_acceptance_rate_tracks_draft_quality():
    """A good draft is accepted often; a bad one rarely."""
    rng = random.Random(0)
    V, K = 4, 3

    def run(q_row, trials=3000):
        p = torch.full((K + 1, V), 1.0 / V)
        q = q_row.unsqueeze(0).repeat(K, 1)
        st = SpeculativeStats()
        for _ in range(trials):
            toks = [draw(q[0], rng) for _ in range(K)]
            _, n = rejection_sample(p, q, toks, rng)
            st.record(K, n)
        return st.acceptance_rate

    good = run(torch.full((V,), 1.0 / V))       # matches target exactly
    bad = run(torch.tensor([0.97, 0.01, 0.01, 0.01]))
    print(f"\n  matched draft:  acceptance {good:.2f}")
    print(f"  mismatched:     acceptance {bad:.2f}")
    assert good > 0.95
    assert bad < good


def test_stats_bookkeeping():
    st = SpeculativeStats()
    st.record(4, 4)
    st.record(4, 0)
    assert st.proposed == 8
    assert st.accepted == 4
    assert st.acceptance_rate == pytest.approx(0.5)
    # 4+1 tokens then 0+1 tokens, over 2 target calls
    assert st.tokens_per_target_call == pytest.approx(3.0)


def test_expected_speedup_shape():
    assert expected_speedup(0.0, 4) == pytest.approx(1.0), \
        "zero acceptance must degrade to ordinary decoding, not below it"
    assert expected_speedup(1.0, 4) == pytest.approx(5.0), \
        "perfect acceptance gives k+1 tokens per target call"

    rows = [(k, expected_speedup(0.7, k)) for k in (1, 2, 4, 8, 16)]
    print(f"\n  acceptance rate 0.7:")
    for k, s in rows:
        print(f"    k={k:>2}  ->  {s:.2f} tokens per target call")
    gain_4_8 = rows[3][1] - rows[2][1]
    gain_1_2 = rows[1][1] - rows[0][1]
    assert gain_4_8 < gain_1_2, "expected diminishing returns as k grows"
    print("\n  \033[2mDiminishing returns: reaching the 8th token needs 8")
    print("  consecutive acceptances. Past k=4 or so you are mostly paying")
    print("  draft cost for tokens you will throw away.\033[0m")


def test_draft_cost_is_charged():
    free = expected_speedup(0.8, 4, draft_cost_ratio=0.0)
    costly = expected_speedup(0.8, 4, draft_cost_ratio=0.2)
    assert costly < free, "running the draft model is not free"
