"""Stage 04 (JAX) - Static batching, right-padded.

The spec:

    app/j04_static_batch.py must define

        static_batch_generate(model, prompts, max_tokens) -> list[list[int]]
        padding_waste(output_lens) -> float

Spec and traps in app/j04_static_batch.py. The one that will get you is
`logits_index`: right-padding means the last position is not everyone's last
real token.
"""

import time

import jax
import pytest

from app.j02_cache import cached_generate
from app.j04_static_batch import padding_waste, static_batch_generate


def test_batched_matches_one_at_a_time(jmodel_exact, prompts):
    """Every row must produce what it would have produced alone.

    Prompts of different lengths on purpose: if a short row silently continues
    from its padding, this is where it shows up.
    """
    model = jmodel_exact
    ps = list(prompts)
    got = static_batch_generate(model, ps, max_tokens=10)
    for i, p in enumerate(ps):
        want = cached_generate(model, p, max_tokens=10)
        assert got[i] == want, (
            f"\nrow {i} (prompt {len(model.encode(p))} tokens, batch max "
            f"{max(len(model.encode(x)) for x in ps)})\n"
            f"alone:   {model.decode(want)!r}\n"
            f"batched: {model.decode(got[i])!r}\n"
            "A short row that reads as fluent but unrelated text is the "
            "logits_index bug: you read logits at T-1 instead of at L-1."
        )


def test_very_ragged_batch(jmodel_exact):
    """A 3-token prompt next to a 60-token one. Same answer either way."""
    model = jmodel_exact
    ps = ["Hello", "The history of computing began " * 12, "def f(x):"]
    got = static_batch_generate(model, ps, max_tokens=8)
    for i, p in enumerate(ps):
        want = cached_generate(model, p, max_tokens=8)
        assert got[i] == want, (
            f"row {i} diverged: {model.decode(got[i])!r} != "
            f"{model.decode(want)!r}"
        )


def test_respects_max_tokens(jmodel_exact):
    out = static_batch_generate(jmodel_exact, ["Count:", "List:"], max_tokens=5)
    assert all(len(o) <= 5 for o in out), [len(o) for o in out]


def test_padding_waste_math():
    assert padding_waste([10, 10, 10]) == pytest.approx(0.0)
    assert padding_waste([5, 10]) == pytest.approx(0.25)
    assert padding_waste([1, 1, 1, 100]) == pytest.approx(1 - 103 / 400)
    assert padding_waste([]) == 0.0
    assert padding_waste([0, 0]) == 0.0


def test_batching_is_nearly_free(jmodel):
    """The reason batching exists: decode is bandwidth-bound.

    Eight sequences read the same weights as one, so eight rows cost barely
    more than one row. Throughput scales; latency does not degrade much.
    """
    model = jmodel
    p = "The history of computing began"
    rows = []
    for B in (1, 4, 8):
        prompts = [p] * B
        static_batch_generate(model, prompts, 4)             # warm this shape
        t = time.perf_counter()
        out = static_batch_generate(model, prompts, 16)
        ms = (time.perf_counter() - t) * 1000
        rows.append((B, ms, sum(len(o) for o in out)))

    print(f"\n  {'batch':>6} {'ms':>9} {'tokens':>8} {'tok/s':>9}")
    for B, ms, n in rows:
        print(f"  {B:>6} {ms:>8.0f} {n:>8} {n / (ms / 1000):>9.1f}")

    tps = [n / (ms / 1000) for _, ms, n in rows]
    assert tps[-1] > tps[0] * 2.5, (
        f"batch 8 got {tps[-1]:.0f} tok/s vs {tps[0]:.0f} at batch 1 -- "
        "batching should be close to free on a bandwidth-bound decode"
    )
    print(f"\n  \033[1m8x the sequences for {rows[-1][1] / rows[0][1]:.1f}x the time.\033[0m")


def test_the_batch_runs_until_its_slowest_member(jmodel):
    """The cost this stage cannot fix, measured.

    One row that finishes at token 3 keeps its slot for the rest of the batch.
    In JAX that slot does not even get cheaper when it goes idle -- the shape
    is fixed, so the compiled step does the same work for a finished row as for
    a live one. Stage 05 is about refilling the slot instead.
    """
    model = jmodel
    ps = [
        model.tokenizer.apply_chat_template(
            [{"role": "user", "content": "Say hi."}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False),
        "The history of computing began",
        "In 1969, humans first",
    ]
    outs = static_batch_generate(model, ps, max_tokens=24)
    lens = [len(o) for o in outs]
    waste = padding_waste(lens)

    print(f"\n  output lengths: {lens}")
    print(f"  \033[1mdecoded token-slots wasted: {waste * 100:.0f}%\033[0m")
    print("  \033[2mEvery wasted slot ran the full model. On this track it is")
    print("  worse than on the torch one: a finished row cannot even shrink")
    print("  the batch, because the batch shape is the compilation.\033[0m")
    assert 0.0 <= waste < 1.0
