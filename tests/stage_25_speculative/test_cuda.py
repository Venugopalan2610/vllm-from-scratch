"""Stage 25 - speculative decoding inside the engine.

The spec is in app/s25_speculative.py. Speculation must be lossless: the
greedy output is the same token for token, and a sampled draw has the target
distribution. Then it must be faster where drafts are right.
"""

from pathlib import Path

import pytest
import torch

from app.s13_sampler import SamplingParams
from app.s22_engine import LLMEngine
from app.s23_graphs import GraphedModelRunner
from app.s25_speculative import (
    SpeculativeEngine,
    target_probs,
    verify,
    verify_greedy,
)
from tests.helpers import CAPSTONE_PROMPTS, elapsed_ms

# A stable file that the repo provides. The model copies it almost exactly.
SOURCE_EXCERPT = (Path(__file__).resolve().parents[2] / "tvllm"
                  / "model.py").read_text()[:1500]
COPY_PROMPT = ("Here is a Python file:\n\n" + SOURCE_EXCERPT
               + "\n\nRepeat the file above exactly, from the start:\n\n")


def test_greedy_verify_stops_at_the_first_miss():
    logits_rows = torch.full((4, 10), -1.0)
    for row, best in enumerate([3, 5, 7, 9]):
        logits_rows[row, best] = 1.0
    assert verify_greedy(logits_rows, [3, 5, 1]) == [3, 5, 7]
    assert verify_greedy(logits_rows, [3, 5, 7]) == [3, 5, 7, 9]
    assert verify_greedy(logits_rows, [4, 5, 7]) == [3]


def test_sampled_verify_keeps_the_target_distribution(device):
    """20,000 draws of the first emitted token, with a draft that is often
    wrong. The histogram must match the target."""
    torch.manual_seed(0)
    logits_rows = torch.randn(2, 6, device=device)
    params = SamplingParams(temperature=1.0)
    target = target_probs(logits_rows[:1], params)[0].cpu()
    generator = torch.Generator(device=device).manual_seed(1)
    counts = torch.zeros(6)
    for _ in range(20_000):
        counts[verify(logits_rows, [2], params, generator)[0]] += 1
    torch.testing.assert_close(counts / counts.sum(), target, atol=0.015, rtol=0)


def _generate(model, engine_class, prompt, max_tokens, **engine_options):
    """One request through a graphed engine.
    -> (the tokens, the wall milliseconds, the engine)."""
    runner = GraphedModelRunner(model, 400, max_model_len=2048,
                                buckets=(1, 2, 4, 8))
    runner.capture()
    engine = engine_class(model, 400, runner=runner, **engine_options)
    engine.add_request(0, prompt, max_tokens, ignore_eos=True)
    outputs, wall_ms = elapsed_ms(engine.run_to_completion)
    runner.kv_caches.clear()
    return outputs[0], wall_ms, engine


@pytest.mark.parametrize("prompt", [COPY_PROMPT, CAPSTONE_PROMPTS[3]])
def test_greedy_output_is_unchanged(nvcc, tmodel_exact, prompt):
    plain, _, _ = _generate(tmodel_exact, LLMEngine, prompt, 60)
    speculative, _, engine = _generate(tmodel_exact, SpeculativeEngine, prompt, 60)
    assert speculative == plain
    assert engine.stats.proposed > 0, "no draft was ever proposed"


def test_max_tokens_is_respected(nvcc, tmodel_exact):
    output, _, _ = _generate(tmodel_exact, SpeculativeEngine, COPY_PROMPT, 37)
    assert len(output) == 37


def test_copying_is_much_faster(nvcc, tmodel):
    """Batch 1 on a copy task, graphs on for both. The n-gram drafts are
    right almost every time."""
    _, plain_ms, _ = _generate(tmodel, LLMEngine, COPY_PROMPT, 200)
    _, speculative_ms, engine = _generate(tmodel, SpeculativeEngine,
                                          COPY_PROMPT, 200)
    stats = engine.stats
    print(f"\n  copy task: {plain_ms / speculative_ms:.2f}x, acceptance "
          f"{stats.acceptance_rate:.2f}, {stats.tokens_per_target_call:.2f} "
          f"tokens for each verify")
    assert plain_ms / speculative_ms >= 1.5


def test_no_draft_costs_nothing(nvcc, tmodel):
    """With no match, there is no draft, and the step is a plain decode."""
    prompt = "Zq7 vex lorp 91 !! qa"
    _, plain_ms, _ = _generate(tmodel, LLMEngine, prompt, 40)
    _, speculative_ms, _ = _generate(tmodel, SpeculativeEngine, prompt, 40)
    assert speculative_ms <= plain_ms * 1.15
