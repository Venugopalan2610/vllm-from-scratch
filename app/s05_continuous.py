"""Stage 05 - continuous batching (scheduling at each iteration).

`./vc lore 5` for the insight. `./vc test 5` to check yourself.

This is the largest throughput idea of this repo, from Orca (OSDI '22). The
batch of stage 04 runs until its SLOWEST row finishes. This batch changes
after every forward pass.
"""

import torch


class Request:
    """One generation in progress. The checks read .id, .output_ids and
    .finished."""

    def __init__(self, rid, prompt_ids, max_tokens):
        self.id = rid
        self.prompt_ids = list(prompt_ids)
        self.max_tokens = max_tokens
        self.output_ids = []
        self.finished = False


class ContinuousEngine:
    """Required attributes (the checks read them):

        .waiting          the queue of Requests that did not start
        .running          the list of Requests in the batch now
        .max_batch_size   the largest number of rows that run at one time
        .steps            the number of FORWARD PASSES (not loop iterations:
                          an iteration that only records and evicts does no
                          GPU work, and it must not count)

    Required methods:

        add_request(rid, prompt: str, max_tokens: int) -> None
        has_work() -> bool
        step() -> list[Request]      # the requests that FINISHED in this step
        run_to_completion() -> {rid: output_ids}
    """

    def __init__(self, model, tokenizer, max_batch_size=8):
        raise NotImplementedError("stage 05: implement ContinuousEngine")

    def add_request(self, rid, prompt, max_tokens):
        raise NotImplementedError

    def has_work(self):
        raise NotImplementedError

    def step(self):
        """One iteration. The checks expect this order:

            1. ADMIT    while len(running) < max_batch_size and a request
                        waits: take it, prefill it alone, and join its KV to
                        the cache of the running batch.
            2. RECORD   each running row has one pending token. Put it into
                        request.output_ids, or mark the request finished
                        (eos, or max_tokens).
            3. EVICT    remove the finished rows from the batch BEFORE the
                        forward pass. That is the difference from stage 04.
            4. FORWARD  one single-token pass for all the rows that remain.

        The invariant: at the start of step(), the pending token of each
        running row is NOT recorded. A prefill and a forward pass each make
        exactly one such token for each row.

        Two traps:

          - Add the column to the attention mask BEFORE the forward pass, not
            after. The model adds one KV entry, so the mask must already
            cover cache_len + 1. If the order is wrong, you get text that
            looks fluent and is garbage after the first correct token. That
            is very difficult to debug.
          - An off-by-one in the record step. If you record the token that
            the forward pass MAKES, not the token that you GAVE it, you lose
            the first generated token of each request, with no error.

        To join a new row to the batch, with left padding:
            cache.layers[i].keys / .values   -> (B, kv_heads, L, head_dim)
            left-pad the shorter one (batch or new row) to the longer length
            torch.cat along dim 0, then DynamicCache([(keys, values), ...])
            cache.batch_select_indices(rows) -> keep only those rows

        Note how much copying that costs. Each admission pads and joins the
        KV tensors of the whole batch again. Stages 06 to 09 replace all of
        it with a pointer update.
        """
        raise NotImplementedError

    def run_to_completion(self):
        """Call step() until has_work() is False. -> {rid: output_ids}."""
        raise NotImplementedError


def run_all(model, tokenizer, jobs, max_batch_size=8):
    """jobs: a list of (rid, prompt, max_tokens). -> {rid: [token ids]}."""
    raise NotImplementedError("stage 05: implement run_all")
