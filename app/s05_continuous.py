"""Stage 05 - continuous batching (iteration-level scheduling).

`./vc lore 5` for the insight. `./vc test 5` to check yourself.

The single biggest throughput idea in this repo, from Orca (OSDI '22). Stage 04's
batch runs until its SLOWEST member finishes. This one re-decides the batch after
every single forward pass.
"""

import torch


class Request:
    """One in-flight generation. Tests read .id, .output_ids and .finished."""

    def __init__(self, rid, prompt_ids, max_tokens):
        self.id = rid
        self.prompt_ids = list(prompt_ids)
        self.max_tokens = max_tokens
        self.output_ids = []
        self.finished = False


class ContinuousEngine:
    """Required attributes (the tests inspect these):

        .waiting          queue of admitted-but-not-started Requests
        .running          list of Requests currently in the batch
        .max_batch_size   how many rows may run at once
        .steps            number of FORWARD PASSES executed (not loop
                          iterations -- an iteration that only harvests and
                          evicts does no GPU work and must not be counted)

    Required methods:

        add_request(rid, prompt: str, max_tokens: int) -> None
        has_work() -> bool
        step() -> list[Request]      # the requests that FINISHED this step
        run_to_completion() -> {rid: output_ids}
    """

    def __init__(self, model, tokenizer, max_batch_size=8):
        raise NotImplementedError("stage 05: implement ContinuousEngine")

    def add_request(self, rid, prompt, max_tokens):
        raise NotImplementedError

    def has_work(self):
        raise NotImplementedError

    def step(self):
        """One iteration. This order is what the tests assume:

            1. ADMIT   -- while len(running) < max_batch_size and waiting:
                          pop a request, prefill it alone, splice its KV into
                          the running batch's cache.
            2. HARVEST -- each running row has one pending token; record it into
                          req.output_ids, or mark the request finished (eos, or
                          max_tokens reached).
            3. EVICT   -- drop finished rows from the batch BEFORE the forward.
                          This is the entire difference from stage 04.
            4. FORWARD -- one single-token pass for every surviving row.

        Invariant that keeps this honest: at the top of step(), every running
        row's pending token has NOT yet been recorded. Both prefill and the
        forward pass produce exactly one such token per row.

        Two traps -- I hit both of these writing the reference solution:

          - Extend the attention mask BEFORE the forward, not after. The model
            is about to append one KV entry, so the mask must already cover
            cache_len + 1. Getting this backwards yields fluent-looking garbage
            after the first correct token, which is miserable to debug.
          - Off-by-one in the harvest. If you record the token produced BY the
            forward rather than the one you fed INTO it, you silently drop the
            first generated token of every request.

        Splicing a new row into the batch, given left-padding:
            cache.layers[i].keys / .values   -> (B, kv_heads, L, head_dim)
            left-pad the shorter of (batch, newcomer) up to the longer length
            torch.cat along dim 0, then DynamicCache([(k, v), ...])
            cache.batch_select_indices(idx)  -> keep only those rows

        Notice how much copying that costs. Every admission re-pads and
        re-concatenates the whole batch's KV tensors. Stages 06-09 replace all
        of it with a pointer update.
        """
        raise NotImplementedError

    def run_to_completion(self):
        """Drive step() until has_work() is False. Returns {rid: output_ids}."""
        raise NotImplementedError


def run_all(model, tokenizer, jobs, max_batch_size=8):
    """jobs: list of (rid, prompt, max_tokens). Returns {rid: [token ids]}."""
    raise NotImplementedError("stage 05: implement run_all")
