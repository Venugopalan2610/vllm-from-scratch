"""Stage 05 (JAX) - Continuous batching over a fixed slot table.

The spec:

    app/j05_continuous.py must define

        Request(rid, prompt_ids, max_tokens)
        ContinuousEngine(model, max_batch_size=8, max_len=512)
            .add_request(rid, prompt, max_tokens)
            .has_work() / .step() / .run_to_completion() / .steps
        run_all(model, jobs, max_batch_size=8) -> {rid: [ids]}

The number this stage is judged on is USEFUL TOKENS PER FORWARD PASS. On a
fixed-shape backend a step costs the same whether one slot is busy or eight, so
occupancy is the entire game.
"""

import pytest

from app.j02_cache import cached_generate
from app.j05_continuous import ContinuousEngine, run_all


def _jobs(prompts, max_tokens):
    return [(i, p, mt) for i, (p, mt) in enumerate(zip(prompts, max_tokens))]


def test_matches_one_at_a_time(jmodel_exact, prompts):
    """Scheduling is not allowed to change anybody's output."""
    model = jmodel_exact
    jobs = _jobs(list(prompts), [10, 10, 10])
    got = run_all(model, jobs, max_batch_size=2, max_len=256)
    assert set(got) == {0, 1, 2}
    for rid, p, mt in jobs:
        want = cached_generate(model, p, max_tokens=mt)
        assert got[rid] == want, (
            f"\nrequest {rid}: {p!r}\n"
            f"alone:      {model.decode(want)!r}\n"
            f"continuous: {model.decode(got[rid])!r}"
        )


def test_more_requests_than_slots(jmodel_exact):
    """Six requests through two slots. Everybody finishes, nobody is lost."""
    model = jmodel_exact
    ps = ["The capital of France is", "def fibonacci(n):", "In 1969, humans",
          "Water boils at", "The largest planet is", "Two plus two is"]
    got = run_all(model, _jobs(ps, [6] * 6), max_batch_size=2, max_len=256)
    assert set(got) == set(range(6)), f"lost requests: {set(range(6)) - set(got)}"
    for rid, p, mt in _jobs(ps, [6] * 6):
        assert got[rid] == cached_generate(model, p, mt), f"request {rid} drifted"


def test_a_freed_slot_is_reused(jmodel_exact):
    """The whole idea: a slot that finishes is refilled, not left idle."""
    model = jmodel_exact
    eng = ContinuousEngine(model, max_batch_size=2, max_len=256)
    eng.add_request("short", "Two plus two is", 2)
    eng.add_request("long", "The history of computing began", 12)
    eng.add_request("waiting", "The capital of France is", 4)

    seen_waiting_running = False
    while eng.has_work():
        eng.step()
        occupied = [eng.slot_req[i] for i in eng.occupied()]
        if any(r.id == "waiting" for r in occupied):
            seen_waiting_running = True
            assert any(r.id == "long" for r in occupied), (
                "the waiting request was admitted, but the long one is gone -- "
                "you evicted a live request instead of a finished one"
            )
    assert seen_waiting_running, (
        "the third request never ran alongside the long one. It should be "
        "admitted the moment the short request frees its slot, not after the "
        "whole batch drains."
    )


def test_no_slot_is_leaked(jmodel_exact):
    """After everything drains, every slot is free."""
    model = jmodel_exact
    eng = ContinuousEngine(model, max_batch_size=3, max_len=256)
    for i, p in enumerate(["Hello", "Water boils at", "The sky is"]):
        eng.add_request(i, p, 4)
    eng.run_to_completion()
    assert eng.occupied() == [], f"slots still held: {eng.occupied()}"
    assert not eng.has_work()


def test_useful_tokens_per_forward_pass(jmodel):
    """The headline number, and the one that is different from torch.

    A static batch runs until its slowest member finishes, so every step after
    the first request completes carries dead rows. Continuous batching refills
    those rows from the queue. On a fixed-shape backend the step costs the same
    either way -- so the win shows up entirely as tokens per forward pass.
    """
    model = jmodel
    ps = ["Hello", "Water boils at", "The capital of France is",
          "The history of computing began", "def fibonacci(n):",
          "In 1969, humans first", "The largest planet is", "Two plus two is"]
    budgets = [2, 4, 6, 24, 4, 20, 3, 2]
    B = 4

    eng = ContinuousEngine(model, max_batch_size=B, max_len=256)
    for rid, (p, mt) in enumerate(zip(ps, budgets)):
        eng.add_request(rid, p, mt)
    got = eng.run_to_completion()
    lens = [len(got[rid]) for rid in range(len(ps))]
    tokens = sum(lens)

    # What a static batch would have cost for the SAME outputs: chunks of B,
    # each chunk running until its slowest member. That is the definition of
    # static batching, so it can be counted rather than re-measured -- and
    # counting it keeps both sides on identical work, which is the only way
    # the comparison means anything.
    static_steps = sum(max(lens[i:i + B]) for i in range(0, len(lens), B))

    static_rate = tokens / max(static_steps, 1)
    cont_rate = tokens / max(eng.steps, 1)
    occupancy = eng.slot_steps / max(eng.steps * B, 1)

    print(f"\n  output lengths: {lens}")
    print(f"\n  {'':<12}{'tokens':>8}{'steps':>8}{'tok/step':>10}")
    print(f"  {'static':<12}{tokens:>8}{static_steps:>8}{static_rate:>10.2f}")
    print(f"  {'continuous':<12}{tokens:>8}{eng.steps:>8}{cont_rate:>10.2f}")
    print(f"\n  slot occupancy: {occupancy * 100:.0f}%  "
          f"(of {B} slots x {eng.steps} steps)")

    assert eng.steps < static_steps, (
        f"continuous took {eng.steps} forward passes, static would have taken "
        f"{static_steps}. Refilling a freed slot has to save steps, or nothing "
        "was refilled."
    )
    assert cont_rate > static_rate * 1.15, (
        f"only {cont_rate:.2f} tok/step vs {static_rate:.2f}. Are finished "
        "slots actually being refilled in the same step they are freed?"
    )
    print(f"\n  \033[1m{cont_rate / static_rate:.2f}x more useful work out of the "
          "same forward pass.\033[0m")
    print("  \033[2mNot faster steps -- the step is a fixed shape and costs")
    print("  what it costs. Fuller ones.\033[0m")
