"""Stage 05 (JAX) - continuous batching over a fixed slot table.

The spec:

    app/j05_continuous.py must define

        Request(rid, prompt_ids, max_tokens)
        ContinuousEngine(model, max_batch_size=8, max_len=512)
            .add_request(rid, prompt, max_tokens)
            .has_work() / .step() / .run_to_completion() / .steps
        run_all(model, jobs, max_batch_size=8) -> {rid: [ids]}

The checks of this stage count USEFUL TOKENS FOR EACH FORWARD PASS. On a
backend with fixed shapes, a step costs the same with one busy slot or with
eight. So the occupancy is the most important thing.
"""

from app.j02_cache import cached_generate
from app.j05_continuous import ContinuousEngine, run_all

SIX_PROMPTS = ["The capital of France is", "def fibonacci(n):",
               "In 1969, humans", "Water boils at", "The largest planet is",
               "Two plus two is"]


def _jobs(prompts, max_tokens_list):
    return [(rid, prompt, max_tokens) for rid, (prompt, max_tokens)
            in enumerate(zip(prompts, max_tokens_list))]


def _assert_same_as_alone(model, jobs, generated):
    assert set(generated) == {rid for rid, _, _ in jobs}, (
        f"lost requests: {({rid for rid, _, _ in jobs}) - set(generated)}")
    for rid, prompt, max_tokens in jobs:
        alone = cached_generate(model, prompt, max_tokens=max_tokens)
        assert generated[rid] == alone, (
            f"\nrequest {rid}: {prompt!r}\n"
            f"alone:      {model.decode(alone)!r}\n"
            f"continuous: {model.decode(generated[rid])!r}")


def test_matches_one_at_a_time(jmodel_exact, prompts):
    """The scheduling must not change the output of any request."""
    jobs = _jobs(list(prompts), [10, 10, 10])
    _assert_same_as_alone(jmodel_exact, jobs,
                          run_all(jmodel_exact, jobs, max_batch_size=2,
                                  max_len=256))


def test_more_requests_than_slots(jmodel_exact):
    """Six requests through two slots. Every request finishes. The engine
    loses none of them."""
    jobs = _jobs(SIX_PROMPTS, [6] * 6)
    _assert_same_as_alone(jmodel_exact, jobs,
                          run_all(jmodel_exact, jobs, max_batch_size=2,
                                  max_len=256))


def test_a_freed_slot_is_reused(jmodel_exact):
    """This is the idea. A new request fills a slot that finishes. The slot
    does not stay idle."""
    engine = ContinuousEngine(jmodel_exact, max_batch_size=2, max_len=256)
    engine.add_request("short", "Two plus two is", 2)
    engine.add_request("long", "The history of computing began", 12)
    engine.add_request("waiting", "The capital of France is", 4)

    waiting_ran_with_long = False
    while engine.has_work():
        engine.step()
        running_ids = {engine.slot_requests[slot].id
                       for slot in engine.occupied()}
        if "waiting" in running_ids:
            waiting_ran_with_long = True
            assert "long" in running_ids, (
                "the waiting request got a slot, but the long one is gone. "
                "You evicted a live request, not a finished one.")
    assert waiting_ran_with_long, (
        "the third request never ran with the long one. It must get the slot "
        "at the moment that the short request frees it, not after the whole "
        "batch finishes.")


def test_no_slot_is_leaked(jmodel_exact):
    """After all requests finish, every slot is free."""
    engine = ContinuousEngine(jmodel_exact, max_batch_size=3, max_len=256)
    for rid, prompt in enumerate(["Hello", "Water boils at", "The sky is"]):
        engine.add_request(rid, prompt, 4)
    engine.run_to_completion()
    assert engine.occupied() == [], f"slots still held: {engine.occupied()}"
    assert not engine.has_work()


def test_useful_tokens_per_forward_pass(jmodel):
    """The main number, and the one that is different from torch.

    A static batch runs until its slowest row finishes, so every step after
    the first finished request has dead rows. Continuous batching fills those
    rows from the queue. On a backend with fixed shapes, the step costs the
    same in both cases. So all of the gain shows as tokens for each forward
    pass.
    """
    prompts = ["Hello", "Water boils at", "The capital of France is",
               "The history of computing began", "def fibonacci(n):",
               "In 1969, humans first", "The largest planet is",
               "Two plus two is"]
    batch_size = 4
    engine = ContinuousEngine(jmodel, max_batch_size=batch_size, max_len=256)
    for rid, prompt, max_tokens in _jobs(prompts, [2, 4, 6, 24, 4, 20, 3, 2]):
        engine.add_request(rid, prompt, max_tokens)
    generated = engine.run_to_completion()
    output_lens = [len(generated[rid]) for rid in range(len(prompts))]
    num_tokens = sum(output_lens)

    # The cost of a static batch for the SAME outputs: chunks of batch_size,
    # and each chunk runs until its slowest row. That is the definition of a
    # static batch, so count it, and do not measure it again. A count also
    # keeps the work of both sides the same, so the comparison is fair.
    static_steps = sum(max(output_lens[start:start + batch_size])
                       for start in range(0, len(output_lens), batch_size))
    static_rate = num_tokens / max(static_steps, 1)
    continuous_rate = num_tokens / max(engine.steps, 1)
    occupancy = engine.slot_steps / max(engine.steps * batch_size, 1)

    print(f"\n  output lengths: {output_lens}")
    print(f"\n  {'':<12}{'tokens':>8}{'steps':>8}{'tok/step':>10}")
    print(f"  {'static':<12}{num_tokens:>8}{static_steps:>8}"
          f"{static_rate:>10.2f}")
    print(f"  {'continuous':<12}{num_tokens:>8}{engine.steps:>8}"
          f"{continuous_rate:>10.2f}")
    print(f"\n  slot occupancy: {occupancy * 100:.0f}%  "
          f"(of {batch_size} slots x {engine.steps} steps)")

    assert engine.steps < static_steps, (
        f"continuous took {engine.steps} forward passes, and static takes "
        f"{static_steps}. A freed slot that gets a new request must save "
        "steps. If not, no slot got a new request.")
    assert continuous_rate > static_rate * 1.15, (
        f"only {continuous_rate:.2f} tok/step against {static_rate:.2f}. Does "
        "a finished slot get a new request in the step that frees it?")
    print(f"\n  \033[1m{continuous_rate / static_rate:.2f}x more useful work "
          "from the same forward pass.\033[0m")
    print("  \033[2mNot faster steps: the step has a fixed shape and a")
    print("  fixed cost. Fuller steps.\033[0m")
