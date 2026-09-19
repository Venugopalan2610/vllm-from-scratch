"""Stage 14 - incremental detokenization and stop conditions.

`./vc lore 14` for the insight. `./vc test 14` to check yourself.

This stage is not exciting, and it fixes most of the problems that users
report.

You cannot decode one token at a time and join the pieces. Three reasons:

  1. A BPE token holds leading-space information that has a meaning only in
     context. So a decode of each token alone damages the whitespace.
  2. One emoji or CJK character is frequently several tokens. Decode the
     first one alone, and you get U+FFFD, the replacement character. Users
     see broken characters in the stream that are correct one token later.
  3. A stop STRING can cross a token boundary, so you cannot look for it one
     token at a time. And you must not emit text that later is the start of
     a stop string.
"""


class IncrementalDetokenizer:
    """Required attributes:

        .text          the decoded text so far (cut if a stop string occurred)
        .stopped       True after a stop string
        .stop_reason   the stop string, or None

    Required methods:
        add_token(token_id) -> str   the new SAFE text, frequently ""
        finalize() -> str            the text that is still held back

    The invariant that every check tests:

        "".join(add_token(t) for t in tokens) + finalize()
            ==  tokenizer.decode(tokens)          (up to a stop string)
    """

    def __init__(self, tokenizer, stop_strings=()):
        raise NotImplementedError("stage 14: implement IncrementalDetokenizer")

    def add_token(self, token_id):
        """Two mechanisms, and you need both.

        (a) The prefix offset and read offset method of vLLM. Keep two
            offsets into your token list, and decode two windows that
            overlap:

                prefix_text = decode(tokens[prefix_offset:read_offset])
                new_text    = decode(tokens[prefix_offset:])

            The delta is new_text[len(prefix_text):]. A decode of a window,
            not of one token, keeps the spaces correct.

            If new_text ends with the replacement character, a multi-byte
            character is not complete: return "" and wait. Do NOT move the
            offsets. If you do, you lose the partial bytes.

        (b) Keep back a tail of (longest stop string - 1) characters before
            you emit. If not, you send "STO", then the next token makes it
            "STOP", and the user already has text that you had to hide. You
            cannot take back an SSE frame.

        At a stop string: cut .text at its start, set .stopped, and emit the
        text that remains. The stop string is not part of the output.
        """
        raise NotImplementedError

    def finalize(self):
        """Emit the tail that you kept back. Call it one time, at the end of
        the generation.

        It must also FORCE out a character that is not complete. A generation
        can stop inside a character (max_tokens between the bytes of an
        emoji), and a batch decode gives U+FFFD there. If you keep it for
        ever, your streaming and non-streaming responses do not agree on the
        last character. (The fuzz check finds this. Hand-written cases do
        not.)
        """
        raise NotImplementedError
