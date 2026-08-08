"""Stage 05 (JAX) - continuous batching over a fixed slot table.

`./vc lore 5 --jax` for the insight. `./vc test 5 --jax` to check yourself.

WHAT YOU'RE BUILDING

    class Request(rid, prompt_ids, max_tokens)   .output_ids .finished
    class ContinuousEngine(model, max_batch_size=8, max_len=512)
        .add_request(rid, prompt, max_tokens)
        .has_work()  -> bool
        .occupied()  -> list of busy slot indices
        .step()      -> list[Request] finished this iteration
        .run_to_completion() -> {rid: [token ids]}
        .steps       -> number of FORWARD PASSES executed
        .slot_steps  -> occupied-slot-steps, for the occupancy metric
    run_all(model, jobs, max_batch_size=8, max_len=512) -> {rid: [token ids]}

Schedule per ITERATION, not per request. When a sequence emits eos, free its
slot that same step and admit a waiting request into it.

THE SHAPE OF THE THING, AND WHY IT IS NOT THE TORCH SHAPE

The torch track re-indexes the KV cache when a row finishes, and the batch
tensor gets smaller. You cannot do that, because the batch dimension is baked
into the compiled step -- shrinking it means recompiling, which costs far more
than the row was costing you.

So the batch is a fixed SLOT TABLE. B rows, allocated once, forever:

    cache = model.init_cache(max_batch_size, max_len)     # once, in __init__

    slot 0  [request A, cache_len 137]
    slot 1  [free]
    slot 2  [request C, cache_len 12]
    ...

Admitting means finding a free slot and prefilling into it. Evicting means
marking the slot free -- nothing is copied, nothing is re-indexed, and the next
admission simply overwrites it.

That is a page table with one page per sequence. Stage 06 makes the pages
smaller; the idea is already here.

ADMITTING INTO A SLOT

`model.forward` writes K/V for every row of the batch it is given, so you
cannot prefill one request inside the batched cache without corrupting the
others. Prefill it on its own, then splice:

    scratch = model.init_cache(1, self.max_len)
    logits, (k1, v1) = model.forward(ids[None], cache=scratch, cache_len=[0])

    kc, vc = self.cache
    self.cache = (kc.at[:, slot].set(k1[:, 0]),
                  vc.at[:, slot].set(v1[:, 0]))

This is a real engine's structure: prefill and decode are separate passes, and
the scheduler decides when each happens. Stage 11 is what happens when you stop
letting a long prefill block everybody's decode.

THE INVARIANT THAT MAKES step() CORRECT

At the top of step(), every occupied slot's next_tok holds a token that has NOT
yet been recorded. Both prefill and the decode pass produce exactly one such
pending token per row. So step() is:

    admit waiting requests into free slots   (each prefill leaves a pending token)
    harvest the pending token from every occupied slot
    mark finished requests' slots free
    if nothing is occupied: return
    ONE forward over all B rows, producing the next pending token per row

Idle slots ride along in that forward. They cost you the full model, because
the shape is fixed and there is no such thing as a partial batch. That is the
whole economics of this stage: a step costs the same whether one slot or eight
are busy, so the ONLY thing that matters is keeping slots full.

Which is why the number this stage is judged on is USEFUL TOKENS PER FORWARD
PASS, not wall clock. Stage 12 gives you the other half -- dropping to a
smaller compiled bucket when occupancy really does fall.

TRAPS

  - Idle slots still get a decode step. Keep their cache_len somewhere harmless
    (0) so they never write past max_len, and never harvest from them.

  - `positions` for a decode step is the row's ABSOLUTE position, which is its
    cache_len. Per row, not one scalar for the batch.

  - One host sync per step, not one per row: pull the whole argmax back once,
    then loop over it in Python. And use `np.array(...)`, not `np.asarray(...)`
    -- asarray hands back a READ-ONLY view of the device buffer, and your next
    per-slot write to it raises "assignment destination is read-only".

  - A request longer than max_len must not silently corrupt the cache. Refuse
    it, or size max_len for it.
"""


class Request:
    """One in-flight generation."""

    def __init__(self, rid, prompt_ids, max_tokens):
        raise NotImplementedError("stage 05 (jax): implement Request")


class ContinuousEngine:
    """Iteration-level scheduling over a fixed table of KV slots."""

    def __init__(self, model, max_batch_size=8, max_len=512):
        raise NotImplementedError("stage 05 (jax): implement ContinuousEngine")


def run_all(model, jobs, max_batch_size=8, max_len=512):
    """jobs: list of (rid, prompt, max_tokens). Returns {rid: [token ids]}."""
    raise NotImplementedError("stage 05 (jax): implement run_all")
