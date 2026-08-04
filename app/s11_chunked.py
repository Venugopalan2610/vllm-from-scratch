"""Stage 11 - chunked prefill and mixed batches.

`./vc lore 11` for the insight. `./vc test 11` to check yourself.

Stage 03 measured it: prefill is compute-bound (~34 us/token) and decode is
memory-bound (~7.8 ms/token). One 8000-token prefill is a single enormous step.
Every sequence mid-generation stalls for the whole thing, and users watching a
token stream see it visibly freeze.

The fix, from Sarathi-Serve: give each step a TOKEN BUDGET. A prefill that
exceeds it is split across steps, and the leftover budget is filled with decode
tokens from other sequences. One batch, mixed work.

This is the main throughput-vs-latency dial in every modern serving stack:

    small budget -> smooth streaming, more steps, slightly less throughput
    large budget -> better prefill efficiency, spikier inter-token latency
"""

import math
from collections import deque


class ChunkSeq:
    """Required attributes:

        .id .prompt_len .max_tokens
        .num_computed    prompt tokens prefilled SO FAR (this is the new idea)
        .num_generated   output tokens produced
        .num_tokens      num_computed + num_generated
        .is_prefilling   num_computed < prompt_len
        .done            num_generated >= max_tokens
        .blocks
    """

    def __init__(self, rid, prompt_len, max_tokens):
        raise NotImplementedError("stage 11: implement ChunkSeq")


class ChunkedScheduler:
    """Required attributes:

        .waiting .running .finished .steps
        .token_budget    max tokens of work per step
        .prefill_first   policy flag (see below)

    Required methods:
        add_request(rid, prompt_len, max_tokens)
        has_work() -> bool
        step() -> dict with "prefill", "decoded", "finished", "tokens_used"
                  where "prefill" is a list of (rid, n_tokens_this_chunk)
        run_to_completion()
    """

    def __init__(self, allocator, max_num_seqs=8, token_budget=512,
                 prefill_first=False):
        raise NotImplementedError("stage 11: implement ChunkedScheduler")

    def step(self):
        """One iteration under a token budget.

            budget = token_budget

            DECODES cost 1 token each.
            PREFILL CHUNKS cost min(remaining_budget, prompt_len - num_computed).

            When a sequence's final prefill chunk lands, it also emits its
            first output token -- that is the whole point of prefill.

            "tokens_used" must be the total tokens of work the step did. The
            tests use it as a proxy for how long the step took, which is what
            makes the latency measurement meaningful.

        Policy: `prefill_first` decides who gets the budget first.
            False (default) -- decodes first. Streaming stays smooth, because a
                               decoding sequence is never starved by a prefill.
            True            -- prefills first. Better prefill batching, worse
                               tail latency for everyone already streaming.
        Both are legitimate; production systems expose this as a knob.
        """
        raise NotImplementedError


class UnchunkedScheduler(ChunkedScheduler):
    """Stage 10's behaviour: a prefill must finish in ONE step, however long.

    Implement this too -- it is the baseline the tests measure chunking
    against, and writing both makes the difference concrete.
    """

    def step(self):
        raise NotImplementedError
