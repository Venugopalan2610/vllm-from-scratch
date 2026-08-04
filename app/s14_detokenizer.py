"""Stage 14 - incremental detokenization and stop conditions.

`./vc lore 14` for the insight. `./vc test 14` to check yourself.

The unglamorous stage that fixes most of what users actually complain about.

You cannot decode tokens one at a time and concatenate. Three reasons:

  1. BPE tokens carry leading-space information that only makes sense in
     context, so per-token decoding mangles whitespace.
  2. A single emoji or CJK character is often several tokens. Decode the first
     one alone and you get U+FFFD, the replacement character. Users see
     mojibake mid-stream that "fixes itself" a token later.
  3. A stop STRING can straddle a token boundary, so you cannot check for it
     token by token -- and you must not emit text that later turns out to be
     the start of a stop string.
"""


class IncrementalDetokenizer:
    """Required attributes:

        .text          decoded text so far (truncated if a stop string hit)
        .stopped       whether a stop string has been seen
        .stop_reason   which stop string, or None

    Required methods:
        add_token(token_id) -> str   newly SAFE text; often ""
        finalize() -> str            flush whatever is still held back

    The invariant every test checks:

        "".join(add_token(t) for t in tokens) + finalize()
            ==  tokenizer.decode(tokens)          (up to any stop truncation)
    """

    def __init__(self, tokenizer, stop_strings=()):
        raise NotImplementedError("stage 14: implement IncrementalDetokenizer")

    def add_token(self, token_id):
        """Two mechanisms, and you need both.

        (a) The prefix/read offset trick, which is what vLLM does. Keep two
            offsets into your token list and decode two overlapping windows:

                prefix_text = decode(tokens[prefix_offset:read_offset])
                new_text    = decode(tokens[prefix_offset:])

            The delta is new_text[len(prefix_text):]. Decoding a window rather
            than a single token is what preserves spacing.

            If new_text ends with the replacement character, a multi-byte
            codepoint is still incomplete: return "" and wait. Do NOT advance
            the offsets, or you will lose the partial bytes.

        (b) Hold back a tail of (max_stop_len - 1) characters before emitting.
            Otherwise you stream out "STO", then discover the next token made
            it "STOP", and you have already sent the user text you were
            supposed to suppress. You cannot un-send an SSE frame.

        On hitting a stop string: truncate .text at its start (the stop string
        itself is not part of the output), set .stopped, and emit whatever is
        left over.
        """
        raise NotImplementedError

    def finalize(self):
        """Emit the held-back tail. Call once, when generation ends."""
        raise NotImplementedError
