"""Stage 17 - speculative decoding: draft, verify, reject.

`./vc lore 17` for the insight. `./vc test 17` to check yourself.

Stage 03 measured why this works: decode is memory-bound, so a forward pass
over K tokens costs almost the same as a forward pass over 1. The weights get
read once either way.

So: propose K tokens cheaply, verify all K in ONE target forward pass, and
accept the longest correct prefix.

The remarkable part is that this is LOSSLESS. Modified rejection sampling makes
the emitted distribution exactly the target model's, no matter how bad the
draft is. A bad draft costs you speed, never quality.
"""

import torch


def ngram_propose(tokens, k, n=3):
    """Propose k tokens with no model at all.

    Find the most recent EARLIER occurrence of the last n tokens, and copy
    whatever followed it. Return [] if there is no match.

    Absurdly cheap, and it works well on the things people actually do with
    LLMs -- code, structured output, quoting the prompt back, long documents
    with repeated phrasing.
    """
    raise NotImplementedError("stage 17: implement ngram_propose")


def rejection_sample(target_probs, draft_probs, draft_tokens, rng):
    """Modified rejection sampling. This is the whole algorithm.

        target_probs (K+1, V)   target distribution at each position
        draft_probs  (K,   V)   draft distribution at each position
        draft_tokens (K,)       what the draft proposed
        rng                     random.Random, for reproducibility

    For each proposed token t at position i:

        accept with probability  min(1, p_i(t) / q_i(t))

        if REJECTED: draw a replacement from the normalized residual
                     max(0, p_i - q_i), emit it, and STOP -- everything after
                     this point was conditioned on a token that did not survive.

    If all K are accepted, draw one BONUS token from target_probs[K]. That
    bonus is why K accepted drafts yield K+1 tokens from one target call.

    Returns (emitted_tokens, num_accepted).

    Why the residual is not optional: accepting with min(1, p/q) alone
    under-samples tokens the draft thinks are unlikely. Sampling rejections
    from max(0, p-q) puts exactly that missing mass back. Drop it, or
    renormalize p instead, and you get a subtly wrong distribution that no
    unit test but a statistical one will catch.
    """
    raise NotImplementedError("stage 17: implement rejection_sample")


class SpeculativeStats:
    """Required: .proposed .accepted .rounds .target_calls
                 .acceptance_rate
                 .tokens_per_target_call
        record(n_proposed, n_accepted)

    tokens_per_target_call is the number that actually predicts your speedup:
    every round emits (accepted + 1) tokens for one expensive forward pass.
    """

    def __init__(self):
        raise NotImplementedError("stage 17: implement SpeculativeStats")


def expected_speedup(acceptance_rate, k, draft_cost_ratio=0.0):
    """Expected tokens per target call at a given per-token acceptance rate a.

        sum(a**i for i in 1..k) + 1

    The +1 is the guaranteed token (a correction, or the bonus). Divide by
    (1 + draft_cost_ratio * k) to charge yourself for running the draft model.

    Note the diminishing returns: at a=0.7, going from k=4 to k=8 buys very
    little, because reaching the 8th token requires 8 consecutive acceptances.
    """
    raise NotImplementedError("stage 17: implement expected_speedup")
