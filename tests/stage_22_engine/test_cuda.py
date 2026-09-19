"""Stage 22 - one engine from the parts.

The spec is in app/s22_engine.py. The checks compare tokens in fp32 with HF.
"""

import pytest

from app.s13_sampler import SamplingParams
from app.s22_engine import LLMEngine
from tests.helpers import CAPSTONE_PROMPTS, hf_greedy

OUTPUT_LENS = [30, 8, 45, 20, 60, 12]


@pytest.fixture(scope="module")
def reference(hf_exact):
    model, tokenizer = hf_exact
    return {rid: hf_greedy(model, tokenizer, prompt, num_tokens)
            for rid, (prompt, num_tokens)
            in enumerate(zip(CAPSTONE_PROMPTS, OUTPUT_LENS))}


def run_capstone_load(engine):
    for rid, (prompt, num_tokens) in enumerate(zip(CAPSTONE_PROMPTS,
                                                   OUTPUT_LENS)):
        engine.add_request(rid, prompt, num_tokens, ignore_eos=True)
    return engine.run_to_completion()


def all_events(engine):
    """Step until the engine has no work. -> all the (rid, delta, finished)
    events."""
    events = []
    while engine.has_work():
        events += engine.step()
    return events


def assert_matches_reference(outputs, reference):
    for rid, expected in reference.items():
        assert outputs[rid] == expected, CAPSTONE_PROMPTS[rid]


def test_concurrent_requests_match_hf(nvcc, tmodel_exact, reference):
    """Six requests with different lengths, all at one time."""
    engine = LLMEngine(tmodel_exact, 256, max_num_seqs=8, token_budget=64)
    assert_matches_reference(run_capstone_load(engine), reference)


def test_preemption_changes_no_token(nvcc, tmodel_exact, reference):
    """Six blocks for six requests that need up to five each. The engine must
    preempt, compute again, and still give exactly the same tokens."""
    engine = LLMEngine(tmodel_exact, 6, max_num_seqs=8, token_budget=64,
                       prefix_cache_blocks=0)
    outputs = run_capstone_load(engine)
    assert engine.num_preemptions > 0, (
        "the pool was large enough: no preemption occurred")
    assert_matches_reference(outputs, reference)


def test_no_block_leaks(nvcc, tmodel_exact):
    engine = LLMEngine(tmodel_exact, 64, max_num_seqs=4, token_budget=32)
    run_capstone_load(engine)
    assert not engine.scheduler.running and not engine.scheduler.waiting
    engine.prefix_cache.evict_all()
    assert engine.allocator.num_free == 64


def test_token_budget_is_never_exceeded(nvcc, tmodel_exact):
    engine = LLMEngine(tmodel_exact, 256, max_num_seqs=8, token_budget=16)
    execute = engine.runner.execute
    step_tokens = []

    def counting_execute(chunks):
        step_tokens.append(sum(len(chunk.token_ids) for chunk in chunks))
        return execute(chunks)

    engine.runner.execute = counting_execute
    run_capstone_load(engine)
    assert max(step_tokens) <= 16
    assert max(step_tokens) == 16, (
        "the budget was never full. Is the prefill in chunks?")


def test_prefix_cache_skips_the_shared_prompt(nvcc, tmodel_exact):
    """The second user with the same long system prompt computes almost none
    of it again."""
    system_prompt = "You are a careful assistant who answers briefly. " * 30
    engine = LLMEngine(tmodel_exact, 512, max_num_seqs=8, token_budget=256)
    engine.add_request("a", system_prompt + "Question: what is 2+2?", 4)
    engine.run_to_completion()
    first_prefill = engine.prefill_tokens
    engine.add_request("b", system_prompt + "Question: what is 3+3?", 4)
    engine.run_to_completion()
    second_prefill = engine.prefill_tokens - first_prefill
    assert engine.cached_tokens > 0.8 * first_prefill
    assert second_prefill < 0.2 * first_prefill


def test_prefix_cache_changes_no_token(nvcc, tmodel_exact):
    system_prompt = "Answer in one sentence. " * 20

    def two_waves(prefix_cache_blocks):
        engine = LLMEngine(tmodel_exact, 256,
                           prefix_cache_blocks=prefix_cache_blocks)
        for wave in (range(3), range(3, 6)):
            for rid in wave:
                engine.add_request(rid, system_prompt + CAPSTONE_PROMPTS[rid],
                                   10, ignore_eos=True)
            outputs = engine.run_to_completion()
        return outputs

    assert two_waves(0) == two_waves(None)


def test_step_follows_the_stage_15_contract(nvcc, tmodel_exact):
    """(rid, delta, finished). The deltas make the decoded text, and
    finished is True exactly one time, on the last event."""
    engine = LLMEngine(tmodel_exact, 64)
    engine.add_request("x", CAPSTONE_PROMPTS[0], 12, ignore_eos=True)
    events = all_events(engine)
    assert all(rid == "x" and isinstance(delta, str)
               for rid, delta, _ in events)
    assert [finished for _, _, finished in events] == (
        [False] * (len(events) - 1) + [True])
    text = "".join(delta for _, delta, _ in events)
    assert text == tmodel_exact.tokenizer.decode(engine.output("x"))


def test_stop_string_ends_the_request(nvcc, tmodel_exact):
    engine = LLMEngine(tmodel_exact, 64)
    engine.add_request("x", "Count: 1, 2, 3, 4,", 40, stop=["7"])
    text = "".join(delta for _, delta, _ in all_events(engine))
    assert "7" not in text and len(engine.output("x")) < 40


def test_abort_frees_every_block(nvcc, tmodel_exact):
    engine = LLMEngine(tmodel_exact, 64, prefix_cache_blocks=0)
    engine.add_request("x", CAPSTONE_PROMPTS[0], 200, ignore_eos=True)
    for _ in range(5):
        engine.step()
    assert engine.allocator.num_free < 64
    engine.abort("x")
    assert engine.allocator.num_free == 64 and not engine.has_work()


def test_an_impossible_request_is_rejected(nvcc, tmodel_exact):
    """If not, it waits for ever for blocks that do not exist."""
    engine = LLMEngine(tmodel_exact, 3)
    with pytest.raises(ValueError):
        engine.add_request("big", "hello " * 100, 5)


def test_seeded_sampling_is_reproducible(nvcc, tmodel_exact):
    def sampled_tokens(seed):
        engine = LLMEngine(tmodel_exact, 64)
        engine.add_request("x", CAPSTONE_PROMPTS[5], 20, ignore_eos=True,
                           params=SamplingParams(temperature=1.0, seed=seed))
        return engine.run_to_completion()["x"]

    assert sampled_tokens(7) == sampled_tokens(7)
    assert sampled_tokens(7) != sampled_tokens(8)


def test_last_step_counts_the_work(nvcc, tmodel_exact):
    """Stage 28 makes its roofline from this pair."""
    engine = LLMEngine(tmodel_exact, 64)
    prompt_ids = tmodel_exact.tokenizer(CAPSTONE_PROMPTS[0]).input_ids
    prompt_len = len(prompt_ids)
    engine.add_request("x", prompt_ids, 5, ignore_eos=True)
    engine.step()
    assert (engine.last_step.tokens,
            engine.last_step.context_tokens) == (prompt_len, prompt_len)
    engine.step()
    assert (engine.last_step.tokens,
            engine.last_step.context_tokens) == (1, prompt_len + 1)
