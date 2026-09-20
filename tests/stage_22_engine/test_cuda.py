"""Stage 22 - one engine from the parts.

The spec is in app/s22_engine.py. The checks compare tokens in fp32 with HF.
"""

import pytest

from app.s13_sampler import SamplingParams
from app.s22_engine import LLMEngine, auto_num_blocks
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


def test_batch_invariance_is_a_known_gap(nvcc, tmodel):
    """The same prompt, same seed, at batch 1 and batch 32. Floating-point
    addition is not associative: a different batch size changes the reduction
    tree inside a matmul, which changes rounding, which can flip the argmax.
    This test documents the problem. It does NOT gate on equality, because
    that would require deterministic reductions that torch does not guarantee.
    If the tokens differ, this test prints why. That print is the lesson."""
    prompt = CAPSTONE_PROMPTS[0]
    # Batch 1: alone
    engine1 = LLMEngine(tmodel, 128)
    engine1.add_request("solo", prompt, 20, ignore_eos=True)
    solo = engine1.run_to_completion()["solo"]
    # Batch 32: the same prompt buried in 31 others
    engine32 = LLMEngine(tmodel, 512, max_num_seqs=32, token_budget=512)
    engine32.add_request("target", prompt, 20, ignore_eos=True)
    for i in range(31):
        engine32.add_request(f"pad{i}", CAPSTONE_PROMPTS[i % 6], 20,
                             ignore_eos=True)
    batch = engine32.run_to_completion()["target"]
    differ = sum(1 for a, b in zip(solo, batch) if a != b)
    rate = differ / len(solo) if solo else 0
    print(f"\n  batch-invariance: {differ}/{len(solo)} tokens differ "
          f"({100 * rate:.0f}%)")
    if differ > 0:
        print("  This is expected. Floating-point addition is not associative.")
        print("  A different batch size changes the reduction tree in matmul,")
        print("  which changes rounding, which can flip the argmax at ~2% of")
        print("  positions. Users report this as 'the same prompt gives")
        print("  different text when the server is busy.' There is no simple fix.")
    # NOT a gate: we assert only that the engine ran without error
    assert len(solo) == 20 and len(batch) == 20


def test_auto_num_blocks_profiles_memory(nvcc, tmodel):
    """Real engines derive the block pool size dynamically from free VRAM.
    This test verifies that auto_num_blocks profiles memory and computes
    a realistic positive block count based on model.kv_bytes_per_token()."""
    num_blocks = auto_num_blocks(tmodel, block_size=16, reserve_bytes=1e9)
    assert isinstance(num_blocks, int)
    assert num_blocks > 0
    assert num_blocks * 16 >= 1024


def test_fork_request_shares_blocks_for_n_greater_than_one(nvcc, tmodel):
    """n > 1 sampling forks the request after prefill so children share
    the prompt's KV blocks via refcounts (stage 09) without recomputing."""
    engine = LLMEngine(tmodel, 64, prefix_cache_blocks=0)
    prompt = "In 1969, humans first walked on the"
    engine.add_request("p", prompt, 10, ignore_eos=True)
    engine.step()
    engine.fork_request("p", "c1", params=SamplingParams(temperature=0.7, seed=1))
    engine.fork_request("p", "c2", params=SamplingParams(temperature=0.7, seed=2))
    for b in engine.seqs["p"].blocks:
        assert engine.allocator.ref_count(b) == 3
    outputs = engine.run_to_completion()
    assert len(outputs["p"]) == 10
    assert len(outputs["c1"]) == 10
    assert len(outputs["c2"]) == 10
    assert engine.allocator.num_free == 64


def test_cache_aware_scheduling_prioritizes_prefix_hits(nvcc, tmodel_exact):
    """Cache-aware scheduling: when multiple requests wait in queue, the
    scheduler admits the request with the longest cached prefix match first,
    rather than strict FCFS. This minimizes TTFT and saves prefill compute."""
    system_prompt = "You are a helpful assistant who gives very concise answers. " * 10
    engine = LLMEngine(tmodel_exact, 256, max_num_seqs=1, token_budget=256)
    engine.add_request("warmup", system_prompt + "What is 1+1?", 2)
    engine.run_to_completion()
    assert engine.blocks.prefix_cache.num_cached > 0

    # Add two requests: first has no cache, second matches the long prefix
    engine.add_request("unrelated_first", "Tell me about medieval architecture in Europe.", 2)
    engine.add_request("cached_second", system_prompt + "What is 2+2?", 2)

    # Scheduler should prioritize cached_second
    engine.step()
    running_rids = [s.rid for s in engine.scheduler.running]
    assert "cached_second" in running_rids, (
        "cache-aware scheduler should admit the request that hits the prefix cache first!"
    )
    engine.run_to_completion()

