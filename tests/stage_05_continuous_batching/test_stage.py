"""Stage 05 - continuous batching (scheduling at each iteration).

The spec is in app/s05_continuous.py. In short:

    ContinuousEngine(model, tokenizer, max_batch_size)
        .waiting  .running  .max_batch_size  .steps
        .add_request(rid, prompt, max_tokens)
        .has_work() -> bool
        .step()     -> list[Request]   # the requests that finished in THIS step
        .run_to_completion() -> {rid: output_ids}

    run_all(model, tokenizer, jobs, max_batch_size) -> {rid: output_ids}
        jobs = [(rid, prompt, max_tokens), ...]

The rule of this stage: a finished sequence leaves the batch in the step
where it finishes. A waiting sequence then takes its row at once.
"""

from app.s02_cache import cached_generate
from app.s04_static_batch import static_batch_generate
from app.s05_continuous import ContinuousEngine, run_all
from tests.helpers import elapsed_ms

PROMPT = "The history of computing began"


def _engine_with(model, tokenizer, max_batch_size, max_tokens_list):
    engine = ContinuousEngine(model, tokenizer, max_batch_size=max_batch_size)
    for rid, max_tokens in enumerate(max_tokens_list):
        engine.add_request(rid, PROMPT, max_tokens)
    return engine


def _static_forward_passes(output_lens, batch_size):
    """The passes of a static batch: each chunk pays for its longest row."""
    return sum(max(output_lens[start:start + batch_size])
               for start in range(0, len(output_lens), batch_size))


def test_matches_sequential_at_every_batch_size(hf_exact, prompts, device):
    """Correctness first, and at max_batch_size=1 and 2.

    Those sizes force the admission of requests into a batch that already
    runs. The bugs of the cache join are there.
    """
    model, tokenizer = hf_exact
    expected = {rid: cached_generate(model, tokenizer, prompt, max_tokens=15)
                for rid, prompt in enumerate(prompts)}
    jobs = [(rid, prompt, 15) for rid, prompt in enumerate(prompts)]

    for max_batch_size in (1, 2, 4):
        generated = run_all(model, tokenizer, jobs,
                            max_batch_size=max_batch_size)
        for rid, prompt in enumerate(prompts):
            assert generated[rid] == expected[rid], (
                f"\nmax_batch_size={max_batch_size}, prompt {prompt!r}\n"
                f"sequential: {tokenizer.decode(expected[rid])!r}\n"
                f"engine:     {tokenizer.decode(generated[rid])!r}\n"
                "An admission into a running batch corrupts it when the mask "
                "gets its column after the forward pass, not before.")


def test_max_tokens_is_respected_per_request(hf_exact, device):
    """Each request has its OWN budget. That makes the batch ragged."""
    model, tokenizer = hf_exact
    generated = run_all(model, tokenizer, [(0, PROMPT, 3), (1, PROMPT, 7),
                                           (2, PROMPT, 11)], max_batch_size=4)
    assert [len(generated[rid]) for rid in range(3)] == [3, 7, 11]
    # The prompt is the same, so each output is a prefix of the next.
    assert generated[1][:3] == generated[0]
    assert generated[2][:7] == generated[1]


def test_step_returns_finished_and_evicts_them(hf, device):
    """A finished request must leave .running in the step where it
    finishes."""
    model, tokenizer = hf
    engine = _engine_with(model, tokenizer, 4, (2, 5, 9))
    finished_ids = set()
    while engine.has_work():
        for request in engine.step():
            assert request.finished, (
                f"request {request.id} was returned but .finished is False")
            assert request not in engine.running, (
                f"request {request.id} finished but is still in .running. "
                "Evict before the next forward pass. That is the stage.")
            finished_ids.add(request.id)
    assert finished_ids == {0, 1, 2}


def test_never_exceeds_max_batch_size(hf, device):
    model, tokenizer = hf
    engine = _engine_with(model, tokenizer, 2, [4] * 6)
    while engine.has_work():
        engine.step()
        assert len(engine.running) <= 2, (
            f"the batch grew to {len(engine.running)}, the limit is 2")


def test_waiting_requests_backfill_freed_slots(hf, device):
    """The point of the stage, counted in forward passes.

    `.steps` must count FORWARD PASSES, not loop iterations. An iteration
    that only records and evicts does no GPU work, and it must not count.

    The load: three long requests between short ones. A static batch does
    them in fixed chunks of 4, so every chunk pays for its longest row.
    Continuous batching moves the short ones through the rows that the
    finished ones free.
    """
    model, tokenizer = hf
    output_lens = [30, 4, 4, 4, 30, 4, 4, 4, 30, 4, 4, 4]
    batch_size = 4
    engine = _engine_with(model, tokenizer, batch_size, output_lens)
    while engine.has_work():
        engine.step()

    static_passes = _static_forward_passes(output_lens, batch_size)
    lower_bound = max(max(output_lens), -(-sum(output_lens) // batch_size))
    print(f"\n  output lengths: {output_lens}   (batch size {batch_size})")
    print(f"  {sum(output_lens)} tokens in total")
    print(f"\n  lower bound (perfect packing): {lower_bound:>3} forward passes")
    print(f"  static, no backfill:           {static_passes:>3} forward passes")
    print(f"  \033[1myour engine:                   {engine.steps:>3} forward "
          "passes\033[0m")
    print(f"\n  \033[1m{static_passes / engine.steps:.2f}x fewer passes than "
          "static\033[0m")
    assert engine.steps < static_passes * 0.6, (
        f"{engine.steps} forward passes against {static_passes} for static "
        "batching. The freed rows do not get requests from .waiting.")
    assert engine.steps >= lower_bound, (
        f"{engine.steps} passes is below the lower bound of {lower_bound}. "
        "One sequence can emit only one token in each forward pass, so your "
        "count has a bug.")


def test_requests_can_arrive_after_the_engine_is_running(hf_exact, device):
    """A real server gets requests during a generation. Correctness must
    stay."""
    model, tokenizer = hf_exact
    engine = ContinuousEngine(model, tokenizer, max_batch_size=4)
    engine.add_request(0, "The capital of France is", 15)
    for _ in range(5):
        engine.step()
    engine.add_request(1, "In 1969, humans first", 15)   # joins a running batch

    outputs = {}
    while engine.has_work():
        for request in engine.step():
            outputs[request.id] = request.output_ids

    assert outputs[0] == cached_generate(model, tokenizer,
                                         "The capital of France is", 15)
    assert outputs[1] == cached_generate(model, tokenizer,
                                         "In 1969, humans first", 15), (
        "a request that joined a running batch made the wrong tokens. Look "
        "at its position_ids and at its row of the attention mask.")


def test_beats_static_batching_on_skewed_output_lengths(hf, device):
    """The result. Real traffic has very uneven output lengths."""
    model, tokenizer = hf
    output_lens = [4, 6, 5, 80, 4, 7, 5, 6, 90, 4, 5, 6]
    batch_size = 4
    static_batch_generate(model, tokenizer, ["warm"], 2)

    def run_static():
        for start in range(0, len(output_lens), batch_size):
            chunk = output_lens[start:start + batch_size]
            static_batch_generate(model, tokenizer, [PROMPT] * len(chunk),
                                  max_tokens=max(chunk))

    _, static_ms = elapsed_ms(run_static)
    _, continuous_ms = elapsed_ms(lambda: run_all(
        model, tokenizer,
        [(rid, PROMPT, max_tokens) for rid, max_tokens in enumerate(output_lens)],
        max_batch_size=batch_size))

    print(f"\n  output lengths: {output_lens}")
    print(f"  static  (chunks of {batch_size}): {static_ms:7.0f} ms   about "
          f"{_static_forward_passes(output_lens, batch_size)} forward passes")
    print(f"  continuous:               {continuous_ms:7.0f} ms   about "
          f"{max(output_lens)}+ forward passes")
    print(f"\n  \033[1mSpeedup: {static_ms / continuous_ms:.2f}x\033[0m")
    assert continuous_ms < static_ms / 1.2, (
        f"expected a clear gain. Got {static_ms / continuous_ms:.2f}x.")
    print("\n  \033[2mThe theoretical limit here is about 2x. You do not get")
    print("  it, because each admission pads and copies the KV cache of the")
    print("  whole batch again. Stages 06 to 09 replace that copy with a")
    print("  pointer.\033[0m")
