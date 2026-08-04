"""Reference solution, stage 17 - speculative decoding."""

import torch


def ngram_propose(tokens, k, n=3):
    """Propose k tokens by finding the most recent earlier occurrence of the
    last n tokens and copying what followed it. Returns [] if no match."""
    if len(tokens) < n + 1:
        return []
    pattern = list(tokens[-n:])
    # search backwards, skipping the suffix itself
    for start in range(len(tokens) - n - 1, -1, -1):
        if list(tokens[start:start + n]) == pattern:
            nxt = tokens[start + n: start + n + k]
            if nxt:
                return list(nxt)
    return []


def _sample_from(probs, rng):
    r = rng.random()
    c = 0.0
    for i, p in enumerate(probs.tolist()):
        c += p
        if r < c:
            return i
    return len(probs) - 1


def rejection_sample(target_probs, draft_probs, draft_tokens, rng):
    """Modified rejection sampling. Lossless: the emitted distribution is
    exactly the target's.

        target_probs (K+1, V)   target model's distribution at each position
        draft_probs  (K,   V)   draft model's distribution at each position
        draft_tokens (K,)       what the draft proposed

    Returns (emitted_tokens, num_accepted).
    """
    K = len(draft_tokens)
    out = []
    for i in range(K):
        t = int(draft_tokens[i])
        p = float(target_probs[i, t])
        q = float(draft_probs[i, t])
        if q > 0 and rng.random() < min(1.0, p / q):
            out.append(t)
            continue
        residual = (target_probs[i] - draft_probs[i]).clamp(min=0)
        s = float(residual.sum())
        if s <= 0:
            out.append(_sample_from(target_probs[i], rng))
        else:
            out.append(_sample_from(residual / s, rng))
        return out, i
    out.append(_sample_from(target_probs[K], rng))   # bonus token
    return out, K


class SpeculativeStats:
    def __init__(self):
        self.proposed = 0
        self.accepted = 0
        self.rounds = 0
        self.target_calls = 0

    def record(self, n_proposed, n_accepted):
        self.proposed += n_proposed
        self.accepted += n_accepted
        self.rounds += 1
        self.target_calls += 1

    @property
    def acceptance_rate(self):
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def tokens_per_target_call(self):
        """The actual speedup: tokens emitted per expensive forward pass."""
        if not self.target_calls:
            return 0.0
        return (self.accepted + self.rounds) / self.target_calls


def expected_speedup(acceptance_rate, k, draft_cost_ratio=0.0):
    """Tokens per target call, given an acceptance rate.

    With per-token acceptance probability a, the expected number of accepted
    draft tokens before the first rejection is sum_{i=1..k} a^i, plus one
    guaranteed token (the correction or the bonus).
    """
    a = acceptance_rate
    expected = sum(a ** i for i in range(1, k + 1)) + 1
    return expected / (1 + draft_cost_ratio * k)
