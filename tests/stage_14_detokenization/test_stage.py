"""Stage 14 - Incremental detokenization and stop conditions.

Spec in app/s14_detokenizer.py. The core invariant: streaming must produce
exactly what batch decoding produces. Always.
"""

import random

import pytest

from app.s14_detokenizer import IncrementalDetokenizer

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


def stream(tok, ids, **kw):
    d = IncrementalDetokenizer(tok, **kw)
    out = "".join(d.add_token(t) for t in ids)
    return out + d.finalize(), d


@pytest.mark.parametrize("text", CASES)
def test_streaming_equals_batch_decode(hf, text):
    """The invariant. No exceptions, no 'close enough'."""
    _, tok = hf
    ids = tok(text, add_special_tokens=False).input_ids
    got, _ = stream(tok, ids)
    want = tok.decode(ids)
    assert got == want, (
        f"\ninput:    {text!r}\nstreamed: {got!r}\nbatch:    {want!r}"
    )


def test_fuzz_random_token_sequences(hf):
    """Random token ids find the boundary cases handwritten strings miss."""
    _, tok = hf
    rng = random.Random(0)
    vocab = min(tok.vocab_size, 150000)
    for trial in range(60):
        ids = [rng.randrange(vocab) for _ in range(rng.randint(1, 25))]
        got, _ = stream(tok, ids)
        want = tok.decode(ids)
        assert got == want, f"trial {trial}, ids={ids}\n got {got!r}\nwant {want!r}"


def test_naive_per_token_decoding_is_broken(hf):
    """Demonstrates why this stage exists. Not a test of your code."""
    _, tok = hf
    text = "Emoji: 🎉🚀 done"
    ids = tok(text, add_special_tokens=False).input_ids
    naive = "".join(tok.decode([t]) for t in ids)
    correct = tok.decode(ids)
    got, _ = stream(tok, ids)

    print(f"\n  naive per-token: {naive!r}")
    print(f"  correct:         {correct!r}")
    print(f"  yours:           {got!r}")
    assert got == correct
    if naive != correct:
        print("\n  \033[2mThe naive version emits U+FFFD for the first half of a")
        print("  multi-byte codepoint. Users see mojibake that 'repairs itself'")
        print("  one token later -- a very recognisable streaming bug.\033[0m")


def test_no_replacement_characters_are_ever_emitted(hf):
    """Partial UTF-8 must be buffered, never streamed."""
    _, tok = hf
    for text in ("🎉🚀🌟", "日本語", "🇯🇵 flag"):
        ids = tok(text, add_special_tokens=False).input_ids
        d = IncrementalDetokenizer(tok)
        for t in ids:
            chunk = d.add_token(t)
            assert "�" not in chunk, (
                f"emitted a replacement char for {text!r} -- an incomplete "
                "codepoint was flushed instead of buffered"
            )


# ---- stop strings ---------------------------------------------------

def test_stop_string_truncates_output(hf):
    _, tok = hf
    ids = tok("The answer is 42. STOP and more text",
              add_special_tokens=False).input_ids
    got, d = stream(tok, ids, stop_strings=["STOP"])
    assert d.stopped
    assert d.stop_reason == "STOP"
    assert "STOP" not in got, "the stop string itself must not be emitted"
    assert "more text" not in got, "text after the stop string leaked out"
    assert got.startswith("The answer is 42.")


def test_stop_string_spanning_multiple_tokens(hf):
    """The straddling case: the stop string is not a single token."""
    _, tok = hf
    stop = "\n\nHuman:"
    full = "Some reply here." + stop + " next turn"
    ids = tok(full, add_special_tokens=False).input_ids
    assert len(tok(stop, add_special_tokens=False).input_ids) > 1, \
        "test assumes the stop string is multi-token"

    got, d = stream(tok, ids, stop_strings=[stop])
    assert d.stopped, "failed to detect a stop string that spans tokens"
    assert stop not in got
    assert "next turn" not in got


def test_partial_stop_string_is_never_emitted_early(hf):
    """You cannot un-send an SSE frame.

    While "STO" might still become "STOP", it must be held back.
    """
    _, tok = hf
    ids = tok("Wait for it: STOP now", add_special_tokens=False).input_ids
    d = IncrementalDetokenizer(tok, stop_strings=["STOP"])
    emitted = ""
    for t in ids:
        emitted += d.add_token(t)
        assert "STOP" not in emitted
        if d.stopped:
            break
    assert d.stopped


def test_no_stop_string_means_nothing_is_held_back(hf):
    _, tok = hf
    ids = tok("plain text with no stops", add_special_tokens=False).input_ids
    got, d = stream(tok, ids)
    assert not d.stopped
    assert got == tok.decode(ids)


def test_multiple_stop_strings(hf):
    _, tok = hf
    ids = tok("alpha beta END gamma", add_special_tokens=False).input_ids
    got, d = stream(tok, ids, stop_strings=["FINISH", "END", "DONE"])
    assert d.stopped and d.stop_reason == "END"
    assert "gamma" not in got


def test_finalize_flushes_the_held_back_tail(hf):
    """Text held back for stop-string safety must still arrive at the end."""
    _, tok = hf
    text = "the end of this is held back"
    ids = tok(text, add_special_tokens=False).input_ids
    d = IncrementalDetokenizer(tok, stop_strings=["NEVER_APPEARS"])
    during = "".join(d.add_token(t) for t in ids)
    tail = d.finalize()
    assert during + tail == tok.decode(ids), (
        "finalize() must flush the safety buffer, or every response is "
        "silently truncated by a few characters"
    )
    assert tail != "", "expected some text to have been held back"
