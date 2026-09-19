"""Stage 14 - incremental detokenization and stop conditions.

The spec is in app/s14_detokenizer.py. The main invariant: a stream must
give exactly what a batch decode gives. Always.
"""

import random

import pytest

from app.s14_detokenizer import IncrementalDetokenizer

REPLACEMENT_CHARACTER = "�"
CASES = [
    "Hello world, this is a test.",
    "日本語のテキストです",
    "Emoji: 🎉🚀 done",
    "  leading and   multiple   spaces",
    "def f(x):\n    return x+1",
    "café naïve résumé",
    "Ünïcödé ïs härd",
    "mixed 中文 and English 🙂 together",
    "",
    "a",
]


def token_ids_of(tokenizer, text):
    return tokenizer(text, add_special_tokens=False).input_ids


def stream(tokenizer, token_ids, **options):
    """-> (all the streamed text, the detokenizer)."""
    detokenizer = IncrementalDetokenizer(tokenizer, **options)
    streamed = "".join(detokenizer.add_token(token) for token in token_ids)
    return streamed + detokenizer.finalize(), detokenizer


@pytest.mark.parametrize("text", CASES)
def test_streaming_equals_batch_decode(hf, text):
    """The invariant. No exceptions, no "almost the same"."""
    _, tokenizer = hf
    token_ids = token_ids_of(tokenizer, text)
    streamed, _ = stream(tokenizer, token_ids)
    expected = tokenizer.decode(token_ids)
    assert streamed == expected, (
        f"\ninput:    {text!r}\nstreamed: {streamed!r}\nbatch:    {expected!r}")


def test_fuzz_random_token_sequences(hf):
    """Random token ids find the boundary cases that hand-written strings
    miss."""
    _, tokenizer = hf
    rng = random.Random(0)
    vocab_size = min(tokenizer.vocab_size, 150000)
    for trial in range(60):
        token_ids = [rng.randrange(vocab_size)
                     for _ in range(rng.randint(1, 25))]
        streamed, _ = stream(tokenizer, token_ids)
        expected = tokenizer.decode(token_ids)
        assert streamed == expected, (
            f"trial {trial}, ids={token_ids}\n     got {streamed!r}\n"
            f"expected {expected!r}")


def test_naive_per_token_decoding_is_broken(hf):
    """Shows why this stage exists. Not a check of your code."""
    _, tokenizer = hf
    token_ids = token_ids_of(tokenizer, "Emoji: 🎉🚀 done")
    one_at_a_time = "".join(tokenizer.decode([token]) for token in token_ids)
    correct = tokenizer.decode(token_ids)
    streamed, _ = stream(tokenizer, token_ids)

    print(f"\n  one token at a time: {one_at_a_time!r}")
    print(f"  correct:             {correct!r}")
    print(f"  yours:               {streamed!r}")
    assert streamed == correct
    if one_at_a_time != correct:
        print("\n  \033[2mThe simple version emits U+FFFD for the first half of")
        print("  a multi-byte character. Users see broken characters that are")
        print("  correct one token later. That is a well-known streaming")
        print("  bug.\033[0m")


def test_no_replacement_characters_are_ever_emitted(hf):
    """Keep a partial UTF-8 character in the buffer. Never stream it."""
    _, tokenizer = hf
    for text in ("🎉🚀🌟", "日本語", "🇯🇵 flag"):
        detokenizer = IncrementalDetokenizer(tokenizer)
        for token in token_ids_of(tokenizer, text):
            assert REPLACEMENT_CHARACTER not in detokenizer.add_token(token), (
                f"emitted a replacement character for {text!r}. A character "
                "that was not complete went out, and it had to stay in the "
                "buffer.")


# ---- stop strings ---------------------------------------------------

def test_stop_string_truncates_output(hf):
    _, tokenizer = hf
    token_ids = token_ids_of(tokenizer, "The answer is 42. STOP and more text")
    streamed, detokenizer = stream(tokenizer, token_ids, stop_strings=["STOP"])
    assert detokenizer.stopped
    assert detokenizer.stop_reason == "STOP"
    assert "STOP" not in streamed, "the stop string must not be emitted"
    assert "more text" not in streamed, "text after the stop string got out"
    assert streamed.startswith("The answer is 42.")


def test_stop_string_spanning_multiple_tokens(hf):
    """The crossing case: the stop string is not one token."""
    _, tokenizer = hf
    stop = "\n\nHuman:"
    assert len(token_ids_of(tokenizer, stop)) > 1, (
        "this check needs a stop string of several tokens")
    token_ids = token_ids_of(tokenizer, "Some reply here." + stop + " next turn")

    streamed, detokenizer = stream(tokenizer, token_ids, stop_strings=[stop])
    assert detokenizer.stopped, (
        "did not find a stop string that crosses tokens")
    assert stop not in streamed
    assert "next turn" not in streamed


def test_partial_stop_string_is_never_emitted_early(hf):
    """You cannot take back an SSE frame.

    While "STO" can still become "STOP", keep it back.
    """
    _, tokenizer = hf
    detokenizer = IncrementalDetokenizer(tokenizer, stop_strings=["STOP"])
    emitted = ""
    for token in token_ids_of(tokenizer, "Wait for it: STOP now"):
        emitted += detokenizer.add_token(token)
        assert "STOP" not in emitted
        if detokenizer.stopped:
            break
    assert detokenizer.stopped


def test_no_stop_string_means_nothing_is_held_back(hf):
    _, tokenizer = hf
    token_ids = token_ids_of(tokenizer, "plain text with no stops")
    streamed, detokenizer = stream(tokenizer, token_ids)
    assert not detokenizer.stopped
    assert streamed == tokenizer.decode(token_ids)


def test_multiple_stop_strings(hf):
    _, tokenizer = hf
    token_ids = token_ids_of(tokenizer, "alpha beta END gamma")
    streamed, detokenizer = stream(tokenizer, token_ids,
                                   stop_strings=["FINISH", "END", "DONE"])
    assert detokenizer.stopped and detokenizer.stop_reason == "END"
    assert "gamma" not in streamed


def test_finalize_flushes_the_held_back_tail(hf):
    """The text kept back for stop-string safety must still arrive at the
    end."""
    _, tokenizer = hf
    token_ids = token_ids_of(tokenizer, "the end of this is held back")
    detokenizer = IncrementalDetokenizer(tokenizer,
                                         stop_strings=["NEVER_APPEARS"])
    during = "".join(detokenizer.add_token(token) for token in token_ids)
    tail = detokenizer.finalize()
    assert during + tail == tokenizer.decode(token_ids), (
        "finalize() must emit the safety buffer. If not, every response "
        "loses a few characters at the end, with no error.")
    assert tail != "", "expected some text in the buffer at the end"
