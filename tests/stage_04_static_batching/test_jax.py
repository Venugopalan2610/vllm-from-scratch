"""Stage 04 (JAX) - static batching, right-padded.

The spec:

    app/j04_static_batch.py must define

        static_batch_generate(model, prompts, max_tokens) -> list[list[int]]
        padding_waste(output_lens) -> float

The spec and the traps are in app/j04_static_batch.py. The difficult one is
`logits_index`: with right padding, the last position is not the last real
token of every row.
"""

import pytest

from app.j02_cache import cached_generate
from app.j04_static_batch import padding_waste, static_batch_generate
from tests.helpers import elapsed_ms


def _assert_same_as_alone(model, prompts, max_tokens):
    batched = static_batch_generate(model, prompts, max_tokens=max_tokens)
    longest = max(len(model.encode(prompt)) for prompt in prompts)
    for row, prompt in enumerate(prompts):
        alone = cached_generate(model, prompt, max_tokens=max_tokens)
        assert batched[row] == alone, (
            f"\nrow {row} (prompt of {len(model.encode(prompt))} tokens, "
            f"longest {longest})\n"
            f"alone:   {model.decode(alone)!r}\n"
            f"batched: {model.decode(batched[row])!r}\n"
            "A short row with fluent but unrelated text is the logits_index "
            "bug: you read the logits at T-1, not at L-1.")


def test_batched_matches_one_at_a_time(jmodel_exact, prompts):
    """Every row must make what it makes alone.

    The prompts have different lengths on purpose: if a short row continues
    from its padding, this check finds it.
    """
    _assert_same_as_alone(jmodel_exact, list(prompts), 10)


def test_very_ragged_batch(jmodel_exact):
    """A prompt of 3 tokens next to one of 60 tokens. The same answer."""
    _assert_same_as_alone(
        jmodel_exact,
        ["Hello", "The history of computing began " * 12, "def f(x):"], 8)


def test_respects_max_tokens(jmodel_exact):
    outputs = static_batch_generate(jmodel_exact, ["Count:", "List:"],
                                    max_tokens=5)
    assert all(len(output) <= 5 for output in outputs), [
        len(output) for output in outputs]


def test_padding_waste_math():
    assert padding_waste([10, 10, 10]) == pytest.approx(0.0)
    assert padding_waste([5, 10]) == pytest.approx(0.25)
    assert padding_waste([1, 1, 1, 100]) == pytest.approx(1 - 103 / 400)
    assert padding_waste([]) == 0.0
    assert padding_waste([0, 0]) == 0.0


def test_batching_is_nearly_free(jmodel):
    """The reason for batches: decode is bandwidth-bound.

    Eight sequences read the same weights as one, so eight rows cost only a
    little more than one row. The throughput increases, and the latency
    stays almost the same.
    """
    prompt = "The history of computing began"
    rows = []
    for batch_size in (1, 4, 8):
        prompts = [prompt] * batch_size
        static_batch_generate(jmodel, prompts, 4)          # warm this shape
        outputs, batch_ms = elapsed_ms(
            lambda: static_batch_generate(jmodel, prompts, 16))
        rows.append((batch_size, batch_ms,
                     sum(len(output) for output in outputs)))

    print(f"\n  {'batch':>6} {'ms':>9} {'tokens':>8} {'tok/s':>9}")
    for batch_size, batch_ms, num_tokens in rows:
        print(f"  {batch_size:>6} {batch_ms:>8.0f} {num_tokens:>8} "
              f"{num_tokens / (batch_ms / 1000):>9.1f}")

    tokens_per_second = [num_tokens / (batch_ms / 1000)
                         for _, batch_ms, num_tokens in rows]
    assert tokens_per_second[-1] > tokens_per_second[0] * 2.5, (
        f"batch 8 got {tokens_per_second[-1]:.0f} tok/s against "
        f"{tokens_per_second[0]:.0f} at batch 1. A batch must cost almost "
        "nothing on a bandwidth-bound decode.")
    print(f"\n  \033[1m8x the sequences for {rows[-1][1] / rows[0][1]:.1f}x "
          "the time.\033[0m")


def test_the_batch_runs_until_its_slowest_member(jmodel):
    """The cost that this stage cannot fix, measured.

    A row that finishes at token 3 keeps its slot for the rest of the batch.
    In JAX the slot does not even get cheaper when it is idle. The shape does
    not change, so the compiled step does the same work for a finished row
    as for a live one. Stage 05 fills the slot again.
    """
    model = jmodel
    prompts = [
        model.tokenizer.apply_chat_template(
            [{"role": "user", "content": "Say hi."}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False),
        "The history of computing began",
        "In 1969, humans first",
    ]
    output_lens = [len(output) for output in
                   static_batch_generate(model, prompts, max_tokens=24)]
    waste = padding_waste(output_lens)

    print(f"\n  output lengths: {output_lens}")
    print(f"  \033[1mdecode slots wasted: {waste * 100:.0f}%\033[0m")
    print("  \033[2mEach wasted slot ran the full model. On this track it is")
    print("  worse than on the torch track: a finished row cannot make the")
    print("  batch smaller, because the batch shape is the compilation.\033[0m")
    assert 0.0 <= waste < 1.0
