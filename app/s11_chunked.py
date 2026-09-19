"""Stage 11 - chunked prefill and mixed batches.

`./vc lore 11` for the insight. `./vc test 11` to check yourself.

Stage 03 measured it. Prefill is compute-bound, and one prefill token costs
about two hundred times less than one decode step. Decode is memory-bound.
One prefill of 8000 tokens is one very large step. Every sequence in
generation stops for all of it, and a user who reads a token stream sees it
stop.

The fix comes from Sarathi-Serve. Give each step a TOKEN BUDGET. Cut a
prefill that is larger than the budget into parts over several steps. Then
use the rest of the budget for decode tokens of other sequences. One batch,
mixed work.

This is the main control between throughput and latency in every modern
serving system:

    small budget -> smooth streaming, more steps, a little less throughput
    large budget -> better prefill efficiency, a less regular latency
"""

import math
from collections import deque


class ChunkSeq:
    """Required attributes:

        .id .prompt_len .max_tokens
        .num_computed    the prompt tokens prefilled SO FAR (the new idea)
        .num_generated   the output tokens made
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
        .token_budget    the largest number of tokens of work in one step
        .prefill_first   the policy flag (see below)

    Required methods:
        add_request(rid, prompt_len, max_tokens)
        has_work() -> bool
        step() -> dict with "prefill", "decoded", "finished", "tokens_used"
                  where "prefill" is a list of (rid, chunk length)
        run_to_completion()
    """

    def __init__(self, allocator, max_num_seqs=8, token_budget=512,
                 prefill_first=False):
        raise NotImplementedError("stage 11: implement ChunkedScheduler")

    def step(self):
        """One iteration under a token budget.

            budget = token_budget

            A DECODE costs 1 token.
            A PREFILL CHUNK costs min(budget left, prompt_len - num_computed).

            The last prefill chunk of a sequence also emits its first output
            token. That is the purpose of the prefill.

            "tokens_used" must be all the tokens of work that the step did.
            The checks use it as a measure of the time of the step. That
            makes the latency measurement useful.

        Policy: `prefill_first` selects which work gets the budget first.
            False (default)   decodes first. Streaming stays smooth, because
                              a prefill never stops a decoding sequence.
            True              prefills first. Better prefill batches, worse
                              tail latency for all the streams in progress.
        Both are correct. Production systems give a setting for it.
        """
        raise NotImplementedError


class UnchunkedScheduler(ChunkedScheduler):
    """The behavior of stage 10: a prefill must finish in ONE step, at every
    length.

    Write this class too. The checks measure the chunks against it, and with
    both classes the difference is clear.
    """

    def step(self):
        raise NotImplementedError
