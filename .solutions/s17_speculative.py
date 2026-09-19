"""Reference solution, stage 17 - speculative decoding."""


def ngram_propose(tokens, max_draft, ngram=3):
    """Find the last earlier match of the final `ngram` tokens, and propose
    the max_draft tokens after it. -> [] if there is no match."""
    if len(tokens) < ngram + 1:
        return []
    suffix = list(tokens[-ngram:])
    # Search backwards. Skip the suffix itself.
    for start in range(len(tokens) - ngram - 1, -1, -1):
        if list(tokens[start:start + ngram]) == suffix:
            follow_on = tokens[start + ngram:start + ngram + max_draft]
            if follow_on:
                return list(follow_on)
    return []


def sample_from(probs, rng):
    """Inverse-CDF sampling with a Python random.Random."""
    threshold = rng.random()
    cumulative = 0.0
    for token, prob in enumerate(probs.tolist()):
        cumulative += prob
        if threshold < cumulative:
            return token
    return len(probs) - 1


def sample_residual(target_row, draft_row, rng):
    """After a rejection: sample from max(0, p - q), normalized."""
    residual = (target_row - draft_row).clamp(min=0)
    residual_mass = float(residual.sum())
    if residual_mass <= 0:
        return sample_from(target_row, rng)
    return sample_from(residual / residual_mass, rng)


def rejection_sample(target_probs, draft_probs, draft_tokens, rng):
    """Modified rejection sampling. It is lossless: the tokens have exactly
    the distribution of the target.

        target_probs (K+1, V)   the target distribution at each position
        draft_probs  (K,   V)   the draft distribution at each position
        draft_tokens (K,)       the draft tokens

    -> (emitted tokens, number accepted).
    """
    emitted = []
    for position, token in enumerate(int(t) for t in draft_tokens):
        target_prob = float(target_probs[position, token])
        draft_prob = float(draft_probs[position, token])
        if draft_prob > 0 and rng.random() < min(1.0, target_prob / draft_prob):
            emitted.append(token)
            continue
        emitted.append(sample_residual(target_probs[position],
                                       draft_probs[position], rng))
        return emitted, position
    num_drafts = len(draft_tokens)
    emitted.append(sample_from(target_probs[num_drafts], rng))  # bonus token
    return emitted, num_drafts


class SpeculativeStats:
    def __init__(self):
        self.proposed = 0
        self.accepted = 0
        self.rounds = 0
        self.target_calls = 0

    def record(self, num_proposed, num_accepted):
        self.proposed += num_proposed
        self.accepted += num_accepted
        self.rounds += 1
        self.target_calls += 1

    @property
    def acceptance_rate(self):
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def tokens_per_target_call(self):
        """The real speedup: tokens for each expensive forward pass."""
        if not self.target_calls:
            return 0.0
        return (self.accepted + self.rounds) / self.target_calls


def expected_speedup(acceptance_rate, num_draft, draft_cost_ratio=0.0):
    """Tokens for each target call, at one acceptance rate.

    With an acceptance probability a for each token, the expected number of
    draft tokens before the first rejection is sum_{i=1..k} a^i. Add one
    token that always comes (the correction or the bonus).
    """
    expected_tokens = 1 + sum(acceptance_rate ** i
                              for i in range(1, num_draft + 1))
    return expected_tokens / (1 + draft_cost_ratio * num_draft)
