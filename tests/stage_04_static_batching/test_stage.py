"""Stage 04 - static batching with padding.

The spec:

    app/s04_static_batch.py must define

        static_batch_generate(model, tokenizer, prompts, max_tokens)
            -> list[list[int]]       # the generated ids, one list for each prompt
        padding_waste(output_lens: list[int]) -> float

A batch gives the throughput. Stage 03 told you why: the GPU reads the
weights one time for the whole batch.

This stage gets that gain. It also shows you the cost: the padding, and the
wait for the slowest row.
"""

import pytest

from app.s02_cache import cached_generate
from app.s04_static_batch import padding_waste, static_batch_generate
from tests.helpers import elapsed_ms


def test_padding_waste_math():
    assert padding_waste([10, 10, 10]) == pytest.approx(0.0)
    # 3 rows x 100 steps = 300 row-steps, and only 120 of them are useful.
    assert padding_waste([100, 10, 10]) == pytest.approx(1 - 120 / 300)
    assert padding_waste([]) == 0.0
    assert padding_waste([0, 0]) == 0.0


def _assert_same_as_alone(model, tokenizer, prompts, max_tokens, hint):
    batched = static_batch_generate(model, tokenizer, prompts,
                                    max_tokens=max_tokens)
    assert len(batched) == len(prompts)
    for row, prompt in enumerate(prompts):
        alone = cached_generate(model, tokenizer, prompt, max_tokens=max_tokens)
        assert batched[row] == alone, (
            f"\nrow {row}, prompt of {len(tokenizer(prompt).input_ids)} "
            f"tokens\nalone:   {tokenizer.decode(alone)!r}\n"
            f"batched: {tokenizer.decode(batched[row])!r}\n{hint}")


def test_matches_sequential_exactly(hf_exact, prompts, device):
    """The same tokens as a decode of each prompt alone. Padding is no
    excuse."""
    model, tokenizer = hf_exact
    _assert_same_as_alone(model, tokenizer, prompts, 20,
                          "Look at: left padding? position_ids given? the mask "
                          "longer by one column at each step?")


def test_ragged_prompt_lengths(hf_exact, device):
    """Very different prompt lengths in one batch are the point."""
    model, tokenizer = hf_exact
    prompts = ["Hi", "The history of computing began " * 20,
               "In 1969, humans first"]
    _assert_same_as_alone(model, tokenizer, prompts, 12,
                          "A mix of long and short prompts shows the "
                          "left-padding bugs.")


def test_batching_is_faster_than_sequential(hf, device):
    """The throughput gain. Stage 03 predicted it: the same bytes, more
    tokens."""
    model, tokenizer = hf
    prompts = ["The history of computing began"] * 8
    static_batch_generate(model, tokenizer, ["warm"], 4)

    _, sequential_ms = elapsed_ms(lambda: [
        cached_generate(model, tokenizer, prompt, max_tokens=32)
        for prompt in prompts])
    _, batched_ms = elapsed_ms(lambda: static_batch_generate(
        model, tokenizer, prompts, max_tokens=32))

    print(f"\n  sequential (8 runs): {sequential_ms:7.0f} ms")
    print(f"  batched    (1 run):  {batched_ms:7.0f} ms")
    print(f"\n  \033[1mSpeedup: {sequential_ms / batched_ms:.1f}x for a "
          "batch of 8\033[0m")
    print("  \033[2mThe GPU read the weights 8x fewer times. That is all of "
          "the gain.\033[0m")
    assert batched_ms < sequential_ms / 2, (
        "a batch of 8 sequences must be much faster than 8 sequential runs "
        f"({sequential_ms:.0f} ms against {batched_ms:.0f} ms)")


def test_the_bill_padding_waste(hf, device):
    """Now the cost. Real traffic has very uneven output lengths.

    A static batch cannot free the row of a finished sequence. The row stays,
    padded, and uses a batch row and KV memory until the SLOWEST row
    finishes.
    Stage 05 removes this.
    """
    output_lens = [5, 8, 6, 120, 7, 9, 6, 11]   # most answers short, one long
    waste = padding_waste(output_lens)
    print(f"\n  output lengths: {output_lens}")
    print(f"  longest run:    {max(output_lens)} steps x {len(output_lens)} "
          f"rows = {max(output_lens) * len(output_lens)} row-steps")
    print(f"  really used:    {sum(output_lens)}")
    print(f"\n  \033[1mWasted: {waste * 100:.0f}% of your batch "
          "capacity\033[0m")
    assert waste > 0.7
    print("\n  \033[2mYou paid for a batch of 8 and got the throughput of "
          "about 2.")
    print("  Stage 05 evicts a sequence at the moment that it finishes.\033[0m")
