"""Stage 05 - Continuous batching (iteration-level scheduling).

The spec is in app/s05_continuous.py. Summary:

    ContinuousEngine(model, tokenizer, max_batch_size)
        .waiting  .running  .max_batch_size  .steps
        .add_request(rid, prompt, max_tokens)
        .has_work() -> bool
        .step()     -> list[Request]   # finished THIS step
        .run_to_completion() -> {rid: output_ids}

    run_all(model, tokenizer, jobs, max_batch_size) -> {rid: output_ids}
        jobs = [(rid, prompt, max_tokens), ...]

The rule that defines this stage: a finished sequence leaves the batch on the
step it finishes, and a waiting one takes its slot immediately.
"""

import time

import pytest
import torch

from app.s02_cache import cached_generate
from app.s04_static_batch import static_batch_generate
from app.s05_continuous import ContinuousEngine, run_all

P = "The history of computing began"


def test_matches_sequential_at_every_batch_size(hf_exact, prompts, dev):
    """Correctness first, and specifically at max_batch_size=1 and 2.

    Those force requests to be admitted mid-flight, into a batch that is
    already running -- which is where the cache-splicing bugs live.
    """
    model, tok = hf_exact
    want = {i: cached_generate(model, tok, p, max_tokens=15)
            for i, p in enumerate(prompts)}
    jobs = [(i, p, 15) for i, p in enumerate(prompts)]

    for bs in (1, 2, 4):
        got = run_all(model, tok, jobs, max_batch_size=bs)
        for i, p in enumerate(prompts):
            assert got[i] == want[i], (
                f"\nmax_batch_size={bs}, prompt {p!r}\n"
                f"sequential: {tok.decode(want[i])!r}\n"
                f"engine:     {tok.decode(got[i])!r}\n"
                "Mid-flight admission corrupts the batch when the mask is "
                "extended after the forward instead of before."
            )


def test_max_tokens_is_respected_per_request(hf_exact, dev):
    """Each request carries its OWN budget. That is what makes the batch ragged."""
    model, tok = hf_exact
    jobs = [(0, P, 3), (1, P, 7), (2, P, 11)]
    got = run_all(model, tok, jobs, max_batch_size=4)
    assert len(got[0]) == 3
    assert len(got[1]) == 7
    assert len(got[2]) == 11
    # and they must be prefixes of one another, since the prompt is identical
    assert got[1][:3] == got[0]
    assert got[2][:7] == got[1]


def test_step_returns_finished_and_evicts_them(hf, dev):
    """A finished request must leave .running on the step it finishes."""
    model, tok = hf
    eng = ContinuousEngine(model, tok, max_batch_size=4)
    for i, n in enumerate((2, 5, 9)):
        eng.add_request(i, P, n)

    seen = set()
    while eng.has_work():
        finished = eng.step()
        for r in finished:
            assert r.finished, f"request {r.id} returned but .finished is False"
            assert r not in eng.running, (
                f"request {r.id} finished but is still in .running -- "
                "evict before the next forward, that is the whole stage"
            )
            seen.add(r.id)
    assert seen == {0, 1, 2}


def test_never_exceeds_max_batch_size(hf, dev):
    model, tok = hf
    eng = ContinuousEngine(model, tok, max_batch_size=2)
    for i in range(6):
        eng.add_request(i, P, 4)
    while eng.has_work():
        eng.step()
        assert len(eng.running) <= 2, f"batch grew to {len(eng.running)}, cap is 2"


def test_waiting_requests_backfill_freed_slots(hf, dev):
    """The point of the whole stage, counted in forward passes.

    `.steps` must count FORWARD PASSES, not loop iterations -- an iteration that
    only harvests and evicts does no GPU work and must not be counted.

    Workload: three long requests interleaved with short ones. Static batching
    processes them in fixed chunks of 4, so every chunk pays for its longest
    member. Continuous batching streams the short ones through the slots the
    finished ones vacate.
    """
    model, tok = hf
    lens = [30, 4, 4, 4, 30, 4, 4, 4, 30, 4, 4, 4]
    BS = 4

    eng = ContinuousEngine(model, tok, max_batch_size=BS)
    for i, n in enumerate(lens):
        eng.add_request(i, P, n)
    while eng.has_work():
        eng.step()

    no_backfill = sum(max(lens[i:i + BS]) for i in range(0, len(lens), BS))
    lower_bound = max(max(lens), -(-sum(lens) // BS))

    print(f"\n  output lengths: {lens}   (batch size {BS})")
    print(f"  {sum(lens)} tokens total")
    print(f"\n  lower bound (perfect packing): {lower_bound:>3} forward passes")
    print(f"  static, no backfill:           {no_backfill:>3} forward passes")
    print(f"  \033[1myour engine:                   {eng.steps:>3} forward passes\033[0m")
    print(f"\n  \033[1m{no_backfill / eng.steps:.2f}x fewer passes than static\033[0m")

    assert eng.steps < no_backfill * 0.6, (
        f"{eng.steps} forward passes vs {no_backfill} for static batching -- "
        "freed slots are not being backfilled from .waiting"
    )
    assert eng.steps >= lower_bound, (
        f"{eng.steps} passes is below the {lower_bound} lower bound; a single "
        "sequence can only emit one token per forward pass, so this is a bug "
        "in your accounting"
    )


def test_requests_can_arrive_after_the_engine_is_running(hf_exact, dev):
    """Real servers get requests while mid-generation. Correctness must hold."""
    model, tok = hf_exact
    eng = ContinuousEngine(model, tok, max_batch_size=4)
    eng.add_request(0, "The capital of France is", 15)

    for _ in range(5):
        eng.step()

    eng.add_request(1, "In 1969, humans first", 15)  # joins mid-flight

    out = {}
    while eng.has_work():
        for r in eng.step():
            out[r.id] = r.output_ids

    assert out[0] == cached_generate(model, tok, "The capital of France is", 15)
    assert out[1] == cached_generate(model, tok, "In 1969, humans first", 15), (
        "a request admitted into an already-running batch produced the wrong "
        "tokens -- check its position_ids and its row of the attention mask"
    )


def test_beats_static_batching_on_skewed_output_lengths(hf, dev):
    """The payoff. Real traffic has wildly uneven output lengths."""
    model, tok = hf
    lens = [4, 6, 5, 80, 4, 7, 5, 6, 90, 4, 5, 6]
    BS = 4

    static_batch_generate(model, tok, ["warm"], 2)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(0, len(lens), BS):
        chunk = lens[i:i + BS]
        static_batch_generate(model, tok, [P] * len(chunk), max_tokens=max(chunk))
    torch.cuda.synchronize()
    static_ms = (time.perf_counter() - t0) * 1000

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    run_all(model, tok, [(i, P, n) for i, n in enumerate(lens)], max_batch_size=BS)
    torch.cuda.synchronize()
    cont_ms = (time.perf_counter() - t0) * 1000

    static_steps = sum(max(lens[i:i + BS]) for i in range(0, len(lens), BS))
    print(f"\n  output lengths: {lens}")
    print(f"  static  (chunks of {BS}): {static_ms:7.0f} ms   "
          f"~{static_steps} forward passes")
    print(f"  continuous:               {cont_ms:7.0f} ms   "
          f"~{max(lens)}+ forward passes")
    print(f"\n  \033[1mSpeedup: {static_ms / cont_ms:.2f}x\033[0m")

    assert cont_ms < static_ms / 1.2, (
        f"expected a clear win; got {static_ms / cont_ms:.2f}x"
    )
    print("\n  \033[2mThe theoretical ceiling here is ~2x. You will not reach it,")
    print("  because every admission re-pads and re-copies the whole batch's KV")
    print("  cache. That copying is what stages 06-09 replace with a pointer.\033[0m")
