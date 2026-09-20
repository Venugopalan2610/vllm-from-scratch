"""Stage 28 - what your engine does, against what it can do.

The spec is in app/s28_bench.py. The last two checks gate the whole
capstone. One compares your engine with the roofline floor of its own steps.
The other compares it with your stage 05 engine on the same requests. Both
are ratios, measured on your card in the same minute.
"""

import random

import pytest
import torch

import cudalib
from app.s05_continuous import ContinuousEngine
from app.s22_engine import LLMEngine, StepWork
from app.s23_graphs import GraphedModelRunner
from app.s28_bench import (
    Hardware,
    ModelCost,
    model_cost,
    report,
    run_benchmark,
    run_goodput_benchmark,
    step_floor,
)
from tests.helpers import elapsed_ms

WORDS = ("the model reads every weight to make one token and the cache grows "
         "with each step of the decode").split()

EXAMPLE_COST = ModelCost(weight_bytes=1e9, kv_bytes_per_token=1e5,
                         num_params=5e8)
EXAMPLE_CARD = Hardware(read_bandwidth=1e11, flops=1e14)


def _text_requests(num_requests=48, seed=0):
    """-> [(rid, prompt text, max_tokens)], with three prompt lengths and
    three output lengths."""
    rng = random.Random(seed)
    return [(rid, " ".join(rng.choice(WORDS)
                           for _ in range(rng.choice([20, 80, 300]))),
             rng.choice([32, 128, 256]))
            for rid in range(num_requests)]


def _token_requests(tokenizer):
    return [(rid, tokenizer(text).input_ids, max_tokens)
            for rid, text, max_tokens in _text_requests()]


def test_floor_at_batch_one_is_the_weight_read():
    assert step_floor(EXAMPLE_COST, EXAMPLE_CARD, tokens=1,
                      context_tokens=0) == pytest.approx(1e9 / 1e11)


def test_floor_at_a_big_batch_is_the_arithmetic():
    assert step_floor(EXAMPLE_COST, EXAMPLE_CARD, tokens=10_000,
                      context_tokens=0) == pytest.approx(2 * 5e8 * 10_000 / 1e14)


def test_floor_adds_the_kv_reads():
    assert step_floor(EXAMPLE_COST, EXAMPLE_CARD, tokens=4,
                      context_tokens=2000) == pytest.approx(
                          1e9 / 1e11 + 2000 * 1e5 / 1e11)


def test_floor_at_16k_context_flips_roofline_to_kv_bound():
    """At 16k context, KV reads dominate the weight read.
    weight_read = 1e9 / 1e11 = 0.01s
    kv_read = 16384 * 1e5 / 1e11 = 0.016384s > weight_read!
    This is the regime where the roofline argument flips sign."""
    floor_1k = step_floor(EXAMPLE_COST, EXAMPLE_CARD, tokens=1, context_tokens=1024)
    floor_16k = step_floor(EXAMPLE_COST, EXAMPLE_CARD, tokens=1, context_tokens=16384)
    assert floor_16k > 2 * floor_1k
    kv_read_16k = 16384 * EXAMPLE_COST.kv_bytes_per_token / EXAMPLE_CARD.read_bandwidth
    weight_read = EXAMPLE_COST.weight_bytes / EXAMPLE_CARD.read_bandwidth
    assert kv_read_16k > weight_read


class FakeModel:
    class config:
        dtype = torch.bfloat16

    def weight_bytes(self):
        return 1000

    def kv_bytes_per_token(self):
        return 10


class FakeEngine:
    """Two sequences, three steps, known work. No GPU."""

    def __init__(self):
        self.model = FakeModel()
        self.work = [StepWork(8, 8), StepWork(2, 10), StepWork(1, 5)]
        self.num_steps = 0
        self.num_preemptions = 0
        self.outputs = {}

    def add_request(self, rid, prompt_ids, max_tokens, ignore_eos=False):
        assert ignore_eos, "a benchmark must make a fixed number of tokens"
        self.outputs[rid] = [0] * max_tokens

    def has_work(self):
        return self.num_steps < len(self.work)

    def step(self):
        self.last_step = self.work[self.num_steps]
        self.num_steps += 1
        return []

    def output(self, rid):
        return self.outputs[rid]


def test_benchmark_adds_the_floor_of_every_step():
    engine = FakeEngine()
    card = Hardware(read_bandwidth=1e6, flops=1e12)
    result = run_benchmark(engine, [("a", [1], 2), ("b", [1], 1)], card)
    cost = model_cost(engine.model)
    expected_floor = sum(step_floor(cost, card, work.tokens,
                                    work.context_tokens)
                         for work in engine.work)
    assert result["floor_seconds"] == pytest.approx(expected_floor)
    assert result["output_tokens"] == 3 and result["steps"] == 3
    assert result["efficiency"] == pytest.approx(result["floor_seconds"]
                                                 / result["seconds"])
    assert isinstance(report(result), str)


def _this_card():
    return Hardware(cudalib.read_bandwidth(), cudalib.matmul_flops())


def _capstone(model, max_num_seqs, num_blocks=1500):
    """1500 blocks hold 24,000 tokens: 32 sequences of this load with room
    to spare. The checks share the card with three other models."""
    runner = GraphedModelRunner(model, num_blocks, max_model_len=1024)
    runner.capture()
    return LLMEngine(model, num_blocks, runner=runner,
                     max_num_seqs=max_num_seqs, token_budget=1024)


def test_the_engine_reaches_the_roof(nvcc, tmodel):
    """At least 30% of the floor of its own steps. The reference engine
    gets 40% to 50% on a laptop. The gap is the stretch goals."""
    result = run_benchmark(_capstone(tmodel, 32),
                           _token_requests(tmodel.tokenizer), _this_card())
    print("\n  " + report(result))
    assert result["efficiency"] >= 0.30


def test_the_engine_beats_your_stage_05(nvcc, hf, tmodel):
    """The same requests through your stage 05 engine and your capstone.
    This is what the whole course gave you."""
    model, tokenizer = hf
    stage05 = ContinuousEngine(model, tokenizer, max_batch_size=32)
    for rid, text, max_tokens in _text_requests():
        stage05.add_request(rid, text, max_tokens)
    outputs, stage05_ms = elapsed_ms(stage05.run_to_completion)
    stage05_tok_s = sum(len(tokens) for tokens in outputs.values()) / (
        stage05_ms / 1000)
    del stage05

    result = run_benchmark(_capstone(tmodel, 32), _token_requests(tokenizer),
                           _this_card())
    print(f"\n  stage 05: {stage05_tok_s:.0f} tok/s   capstone: "
          f"{result['tok_s']:.0f} tok/s = {result['tok_s'] / stage05_tok_s:.1f}x")
    assert result["tok_s"] >= 2 * stage05_tok_s


def test_the_engine_gates_on_goodput_at_concurrency(nvcc, tmodel):
    """The headline benchmark gate: measure GOODPUT under concurrency at batch 32.
    Batch 1 is a toy; production serving gates on output tokens per second
    that meet TTFT and ITL SLOs."""
    engine = _capstone(tmodel, 32)
    requests = _token_requests(tmodel.tokenizer)
    result = run_goodput_benchmark(engine, requests, _this_card(), ttft_slo=5.0, itl_slo=0.5)
    print("\n  " + report(result))
    assert result["goodput_tok_s"] > 0
    assert result["goodput_ratio"] >= 0.50, (
        f"only {100 * result['goodput_ratio']:.0f}% of tokens met the SLO. "
        "Real serving requires high goodput under concurrency."
    )
