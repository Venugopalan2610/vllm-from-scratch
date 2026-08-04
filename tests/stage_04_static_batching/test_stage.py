"""Stage 04 - Static batching with padding.

The spec:

    app/s04_static_batch.py must define

        static_batch_generate(model, tokenizer, prompts, max_tokens)
            -> list[list[int]]                 # generated ids, one list per prompt
        padding_waste(output_lens: list[int]) -> float

Batching is where throughput comes from (stage 03 told you why: the weights get
read once for the whole batch). This stage gets the win AND shows you the bill
that comes with it -- padding, and waiting on the slowest member.
"""

import time

import pytest
import torch

from app.s02_cache import cached_generate
from app.s04_static_batch import padding_waste, static_batch_generate


def test_padding_waste_math():
    assert padding_waste([10, 10, 10]) == pytest.approx(0.0)
    # 3 slots x 100 steps = 300 slot-steps; only 120 useful
    assert padding_waste([100, 10, 10]) == pytest.approx(1 - 120 / 300)
    assert padding_waste([]) == 0.0
    assert padding_waste([0, 0]) == 0.0


def test_matches_sequential_exactly(hf_exact, prompts, dev):
    """Same tokens as one-at-a-time decoding. No excuses for padding."""
    model, tok = hf_exact
    want = [cached_generate(model, tok, p, max_tokens=20) for p in prompts]
    got = static_batch_generate(model, tok, prompts, max_tokens=20)

    assert len(got) == len(prompts)
    for i, p in enumerate(prompts):
        assert got[i] == want[i], (
            f"\nprompt: {p!r}\n"
            f"sequential: {tok.decode(want[i])!r}\n"
            f"batched:    {tok.decode(got[i])!r}\n"
            "Check: left padding? explicit position_ids? mask extended each step?"
        )


def test_ragged_prompt_lengths(hf_exact, dev):
    """Wildly different prompt lengths in one batch is the whole point."""
    model, tok = hf_exact
    ps = [
        "Hi",
        "The history of computing began " * 20,
        "In 1969, humans first",
    ]
    want = [cached_generate(model, tok, p, max_tokens=12) for p in ps]
    got = static_batch_generate(model, tok, ps, max_tokens=12)
    for i in range(len(ps)):
        assert got[i] == want[i], (
            f"row {i} (prompt {len(tok(ps[i]).input_ids)} tokens) diverged. "
            "Long/short mixes are where left-padding bugs show up."
        )


def test_batching_is_faster_than_sequential(hf, dev):
    """The throughput win. Stage 03 predicted this: same bytes, more tokens."""
    model, tok = hf
    ps = ["The history of computing began"] * 8
    N = 32

    static_batch_generate(model, tok, ["warm"], 4)  # warm kernels

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for p in ps:
        cached_generate(model, tok, p, max_tokens=N)
    torch.cuda.synchronize()
    seq_ms = (time.perf_counter() - t0) * 1000

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    static_batch_generate(model, tok, ps, max_tokens=N)
    torch.cuda.synchronize()
    bat_ms = (time.perf_counter() - t0) * 1000

    print(f"\n  sequential (8 runs): {seq_ms:7.0f} ms")
    print(f"  batched    (1 run):  {bat_ms:7.0f} ms")
    print(f"\n  \033[1mSpeedup: {seq_ms / bat_ms:.1f}x for batch of 8\033[0m")
    print("  \033[2mThe weights were read 8x fewer times. That is the entire win.\033[0m")

    assert bat_ms < seq_ms / 2, (
        f"batching 8 sequences should be far faster than 8 sequential runs "
        f"({seq_ms:.0f} ms vs {bat_ms:.0f} ms)"
    )


def test_the_bill_padding_waste(hf, dev):
    """Now the cost. Real traffic has wildly uneven output lengths.

    A static batch cannot release a finished sequence's slot. It sits there,
    padded, consuming a batch slot and KV memory, until the SLOWEST member of
    the batch is done. This is what stage 05 deletes.
    """
    model, tok = hf
    # a realistic spread: most answers short, one long
    lens = [5, 8, 6, 120, 7, 9, 6, 11]
    waste = padding_waste(lens)
    print(f"\n  output lengths: {lens}")
    print(f"  longest run:    {max(lens)} steps, x {len(lens)} slots = "
          f"{max(lens) * len(lens)} slot-steps")
    print(f"  actually used:  {sum(lens)}")
    print(f"\n  \033[1mWasted: {waste * 100:.0f}% of your batch capacity\033[0m")
    assert waste > 0.7
    print("\n  \033[2mYou paid for a batch of 8 and got the throughput of ~2.")
    print("  Stage 05 evicts finished sequences the instant they finish.\033[0m")
