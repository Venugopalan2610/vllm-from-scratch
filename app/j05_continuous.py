"""Stage 05 (JAX) - continuous batching over a fixed slot table.

`./vc lore 5 --jax` for the insight. `./vc test 5 --jax` to check yourself.

WHAT YOU ARE BUILDING

    class Request(rid, prompt_ids, max_tokens)   .output_ids .finished
    class ContinuousEngine(model, max_batch_size=8, max_len=512)
        .add_request(rid, prompt, max_tokens)
        .has_work()  -> bool
        .occupied()  -> list of busy slot indices
        .slot_requests -> the Request in each slot, or None
        .step()      -> list[Request] finished this iteration
        .run_to_completion() -> {rid: [token ids]}
        .steps       -> number of FORWARD PASSES executed
        .slot_steps  -> occupied-slot-steps, for the occupancy metric
    run_all(model, jobs, max_batch_size=8, max_len=512) -> {rid: [token ids]}

Schedule for each ITERATION, and not for each request. When a sequence emits
eos, free its slot in that same step, and admit a waiting request into it.

THE SHAPE OF THE THING, AND WHY IT IS NOT THE TORCH SHAPE

The torch track indexes the KV cache again when a row finishes, and the batch
tensor becomes smaller. You cannot do that. The batch dimension is part of the
compiled step. A smaller batch means a recompile, and that costs much more
than the idle row costs you.

So the batch is a fixed SLOT TABLE. B rows, allocated one time, forever:

    cache = model.init_cache(max_batch_size, max_len)     # once, in __init__

    slot 0  [request A, cache_len 137]
    slot 1  [free]
    slot 2  [request C, cache_len 12]
    ...

To admit a request, find a free slot and prefill into it. To evict a request,
mark its slot free. You copy nothing, and you index nothing again. The next
admission overwrites the slot.

That is a page table with one page for each sequence. Stage 06 makes the pages
smaller. The idea is already here.

HOW TO ADMIT INTO A SLOT

`model.forward` writes K and V for every row of the batch that it gets. So you
cannot prefill one request inside the batched cache: that corrupts the other
rows. Prefill it alone, then splice:

    scratch_cache = model.init_cache(1, self.max_len)
    logits, (new_keys, new_values) = model.forward(
        prompt_ids[None], cache=scratch_cache, cache_len=[0])

    key_cache, value_cache = self.cache
    self.cache = (key_cache.at[:, slot].set(new_keys[:, 0]),
                  value_cache.at[:, slot].set(new_values[:, 0]))

A real engine has this structure. Prefill and decode are separate passes, and
the scheduler decides when each one happens. Stage 11 shows what happens when a
long prefill stops blocking the decode of everybody else.

THE INVARIANT THAT MAKES step() CORRECT

At the top of step(), every occupied slot holds one pending token that
nobody recorded yet. Prefill and the decode pass both make exactly one such
pending token for each row. So step() is:

    admit waiting requests into free slots   (each prefill leaves a pending token)
    record the pending token of every occupied slot
    mark the slots of finished requests free
    if nothing is occupied: return
    ONE forward over all B rows, which makes the next pending token of each row

Idle slots go through that forward pass. They cost the full model, because
the shape does not change and a partial batch does not exist. That is the
cost model of this stage. A step costs the same with one busy slot or with
eight, so the ONLY important thing is to keep the slots full.

That is why the checks of this stage count USEFUL TOKENS FOR EACH FORWARD PASS,
and not wall clock. Stage 12 gives you the other half: a smaller compiled bucket
when the occupancy really does fall.

TRAPS

  - Idle slots still get a decode step. Keep their cache_len at a harmless
    value (0), so that they never write past max_len. Never record from them.

  - `positions` for a decode step is the ABSOLUTE position of the row, which
    is its cache_len. It is one value for each row, not one scalar for the
    batch.

  - One host sync for each step, not one for each row. Pull the whole argmax
    back one time, then loop over it in Python. Use `np.array(...)`, and not
    `np.asarray(...)`. asarray gives you a READ-ONLY view of the device
    buffer, and your next write to one slot raises "assignment destination is
    read-only".

  - A request longer than max_len must not corrupt the cache with no error.
    Refuse it, or make max_len large enough for it.
"""


class Request:
    """One generation in progress."""

    def __init__(self, rid, prompt_ids, max_tokens):
        raise NotImplementedError("stage 05 (jax): implement Request")


class ContinuousEngine:
    """Scheduling at each iteration, over a fixed table of KV slots."""

    def __init__(self, model, max_batch_size=8, max_len=512):
        raise NotImplementedError("stage 05 (jax): implement ContinuousEngine")


def run_all(model, jobs, max_batch_size=8, max_len=512):
    """jobs: a list of (rid, prompt, max_tokens). -> {rid: [token ids]}."""
    raise NotImplementedError("stage 05 (jax): implement run_all")
