"""Stage 17 - speculative decoding: draft, verify, reject.

`./vc lore 17` for the insight. `./vc test 17` to check yourself.

Stage 03 measured why this works. Decode waits on memory, so a forward pass
over K tokens costs almost as much as a forward pass over 1 token. The GPU
reads the weights one time in both cases.

So: propose K tokens at a low cost, verify all K in ONE forward pass of the
target, and accept the longest correct prefix.

The important part: this is LOSSLESS. Modified rejection sampling makes the
distribution of the output exactly that of the target model, however bad
the draft is. A bad draft costs speed, never quality.

**Limitation.** This stage uses n-gram drafting, which works well on copy
tasks and repetitive text. Real deployments use a draft model, EAGLE, or
MTP heads, and the hard parts are batched speculation with a paged cache,
KV rollback on rejection, and interaction with chunked prefill and
preemption. The 1.5x speedup on a copy task does not predict deployment
gains on diverse traffic.
"""

import torch


def ngram_propose(tokens, max_draft, ngram=3):
    """Propose max_draft tokens with no model.

    Find the last EARLIER match of the final `ngram` tokens, and copy the
    tokens after it. Return [] if there is no match.

    It costs almost nothing. It works well on the usual work of an LLM:
    code, structured output, a quote of the prompt, and long documents with
    repeated phrases.
    """
    raise NotImplementedError("stage 17: implement ngram_propose")


def rejection_sample(target_probs, draft_probs, draft_tokens, rng):
    """Modified rejection sampling. This is the whole algorithm.

        target_probs (K+1, V)   the target distribution at each position
        draft_probs  (K,   V)   the draft distribution at each position
        draft_tokens (K,)       the tokens that the draft proposed
        rng                     random.Random, so that runs repeat

    For each proposed token t at position i:

        accept with probability  min(1, p_i(t) / q_i(t))

        if REJECTED: draw a replacement from the normalized residual
                     max(0, p_i - q_i), emit it, and STOP. Everything
                     after this point depends on a token that was rejected.

    If you accept all K, draw one BONUS token from target_probs[K]. That
    bonus is why K accepted drafts give K+1 tokens from one target call.

    -> (emitted_tokens, num_accepted).

    The residual is necessary. An acceptance with min(1, p/q) alone samples
    too few of the tokens that the draft thinks are improbable. A draw from
    max(0, p-q) after a rejection puts exactly that missing mass back. If
    you drop it, or normalize p again, the distribution is a little wrong.
    Only a statistical check finds that.
    """
    raise NotImplementedError("stage 17: implement rejection_sample")


class SpeculativeStats:
    """Required: .proposed .accepted .rounds .target_calls
                 .acceptance_rate
                 .tokens_per_target_call
        record(num_proposed, num_accepted)

    tokens_per_target_call is the number that predicts your speedup: each
    round emits (accepted + 1) tokens for one expensive forward pass.
    """

    def __init__(self):
        raise NotImplementedError("stage 17: implement SpeculativeStats")


def expected_speedup(acceptance_rate, num_draft, draft_cost_ratio=0.0):
    """The expected tokens for each target call, at an acceptance rate a for
    each token. With k = num_draft:

        sum(a**i for i in 1..k) + 1

    The +1 is the token that always comes (a correction, or the bonus).
    Divide by (1 + draft_cost_ratio * k) to include the cost of the draft
    model.

    Note that the gain decreases: at a=0.7, k=8 gives little more than k=4,
    because the 8th token needs 8 acceptances in sequence.
    """
    raise NotImplementedError("stage 17: implement expected_speedup")


def build_tree_mask(tree_parents, prefix_len=0, device="cpu"):
    """Construct a 2D causal attention mask for tree-based speculative decoding (e.g. Medusa, EAGLE).

    In tree speculation, draft tokens form a tree of candidate hypotheses rather than a
    linear chain. All candidate tokens in the tree are evaluated simultaneously in ONE
    target model forward pass.
    """
    raise NotImplementedError("stage 17: implement build_tree_mask")


def verify_tree_greedy(logits_rows, tree_tokens, tree_parents):
    """Greedy verification for tree speculative decoding."""
    raise NotImplementedError("stage 17: implement verify_tree_greedy")
