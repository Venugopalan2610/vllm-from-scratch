"""Stage 10 - waiting / running queues, admission, and preemption.

`./vc lore 10` for the insight. `./vc test 10` to check yourself.

You have finite KV memory and unbounded demand. Sequences grow one token at a
time, so a batch you legitimately admitted can run out of memory MID-DECODE.
There is no way to avoid this by planning: you cannot know how long a sequence
will be until it emits EOS.

So you need preemption, and you get two choices:

  SWAP      copy the victim's KV blocks out to CPU RAM, copy them back later.
            Costs PCIe bandwidth both ways.
  RECOMPUTE drop the blocks entirely, re-prefill from scratch when readmitted.
            Costs one prefill.

Recompute usually wins, because prefill is compute-bound and fast (stage 03:
~34 us/token) while PCIe is narrow. That is what this stage implements.
"""

import math
from collections import deque


class SeqState:
    """Required attributes:

        .id .prompt_len .max_tokens
        .num_generated      tokens produced so far
        .num_tokens         prompt_len + num_generated (KV footprint)
        .prefilled          has its prompt been computed
        .blocks             physical blocks it owns
        .done               num_generated >= max_tokens
        .preempted_count    how many times it has been kicked out
    """

    def __init__(self, rid, prompt_len, max_tokens):
        raise NotImplementedError("stage 10: implement SeqState")


class Scheduler:
    """Required attributes:

        .waiting  .running  .finished
        .preemptions   total preemptions so far
        .steps         engine iterations that did real work

    Required methods:

        add_request(rid, prompt_len, max_tokens)
        has_work() -> bool
        step() -> dict with keys "prefilled", "decoded", "preempted", "finished"
        run_to_completion() -> list[SeqState]
    """

    def __init__(self, allocator, max_num_seqs=8):
        raise NotImplementedError("stage 10: implement Scheduler")

    def step(self):
        """One engine iteration:

            1. ADMIT from .waiting while
                   len(running) < max_num_seqs
                   AND the allocator can cover the prompt's blocks.
               If a prompt does not fit, STOP admitting -- do not skip ahead to
               a smaller request behind it, or you starve long prompts forever.

            2. DECODE every already-prefilled running sequence: it needs one
               more token, which may need one more block.

            3. If that block is unavailable, PREEMPT. Pick the NEWEST running
               sequence (it has done the least work), free its blocks, reset
               num_generated to 0, mark it un-prefilled, and push it to the
               FRONT of .waiting so it is retried first.

            4. Retire finished sequences and release their blocks.

        Invariants the tests check hard:
          - blocks are never leaked: after everything finishes, the allocator
            is back to full
          - no starvation: every request eventually completes, even when the
            pool is far too small to hold them all at once
          - a preempted sequence restarts from scratch (recompute semantics),
            so it must still end up with exactly max_tokens outputs
        """
        raise NotImplementedError

    def run_to_completion(self, max_steps=100000):
        raise NotImplementedError
