"""Stage 10 - the waiting and running queues, admission, and preemption.

`./vc lore 10` for the insight. `./vc test 10` to check yourself.

You have limited KV memory and unlimited demand. A sequence grows one token
at a time, so a batch that you admitted correctly can run out of memory
DURING decode. A plan cannot prevent this: you do not know the length of a
sequence until it emits EOS.

So you need preemption, and you have two choices:

  SWAP       copy the KV blocks of the victim to CPU RAM, and copy them back
             later. It costs PCIe bandwidth in both directions.
  RECOMPUTE  drop the blocks, and prefill again from the start when the
             sequence comes back. It costs one prefill.

This stage uses recompute. It is not always the cheaper choice. The notebook
part4_adm_2_swapVsRecompute measures both, and on a small model a swap can be
faster. Recompute has a different advantage: you can cut it into chunks and
schedule it (stage 11). A swap-in stops the sequence until its bytes arrive.
"""

import math
from collections import deque


class SeqState:
    """Required attributes:

        .id .prompt_len .max_tokens
        .num_generated      the tokens made so far
        .num_tokens         prompt_len + num_generated (the KV size)
        .prefilled          True after the prefill of the prompt
        .blocks             the physical blocks that it owns
        .done               num_generated >= max_tokens
        .preempted_count    the number of its preemptions
    """

    def __init__(self, rid, prompt_len, max_tokens):
        raise NotImplementedError("stage 10: implement SeqState")


class Scheduler:
    """Required attributes:

        .waiting  .running  .finished
        .preemptions   all the preemptions so far
        .steps         the engine iterations that did real work

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
                   AND the allocator has the blocks of the prompt.
               If a prompt does not fit, STOP the admission. Do not go to a
               smaller request behind it. If you do, a long prompt waits for
               ever.

            2. DECODE each running sequence that is prefilled: it needs one
               more token, and that token can need one more block.

            3. If that block is not available, PREEMPT. Select the NEWEST
               running sequence (it did the least work), free its blocks, set
               num_generated to 0, mark it not prefilled, and put it at the
               FRONT of .waiting, so that it is tried first.

            4. Remove the finished sequences and free their blocks.

        The checks test these invariants hard:
          - no lost blocks: after all requests finish, the allocator is
            full again
          - no request waits for ever: each request completes, also when
            the pool is much too small to hold all of them at one time
          - a preempted sequence starts again from the start (recompute), so
            it must still get exactly max_tokens outputs
        """
        raise NotImplementedError

    def run_to_completion(self, max_steps=100000):
        raise NotImplementedError
