"""Reference solution, stage 14 - incremental detokenization."""


class IncrementalDetokenizer:
    def __init__(self, tokenizer, stop_strings=()):
        self.tok = tokenizer
        self.stops = list(stop_strings)
        self.max_stop = max((len(s) for s in self.stops), default=0)
        self.tokens = []
        self.prefix_offset = 0
        self.read_offset = 0
        self.text = ""        # everything decoded so far (post stop-truncation)
        self.emitted = ""     # everything handed to the caller so far
        self.stopped = False
        self.stop_reason = None

    def _decode_delta(self, force=False):
        """vLLM's prefix/read offset trick.

        Decode two overlapping windows and diff them. If the new window ends in
        a replacement char, a multi-byte codepoint is still incomplete, so emit
        nothing and wait for the next token.
        """
        prefix_text = self.tok.decode(
            self.tokens[self.prefix_offset:self.read_offset],
            skip_special_tokens=False,
        )
        new_text = self.tok.decode(
            self.tokens[self.prefix_offset:],
            skip_special_tokens=False,
        )
        if len(new_text) <= len(prefix_text):
            return ""
        if new_text.endswith("\ufffd") and not force:
            return ""                       # incomplete UTF-8, keep buffering
        delta = new_text[len(prefix_text):]
        self.prefix_offset = self.read_offset
        self.read_offset = len(self.tokens)
        return delta

    def add_token(self, token_id):
        if self.stopped:
            return ""
        self.tokens.append(token_id)
        delta = self._decode_delta()
        if not delta:
            return ""
        self.text += delta

        for s in self.stops:
            idx = self.text.find(s)
            if idx != -1:
                self.stopped = True
                self.stop_reason = s
                self.text = self.text[:idx]
                out = self.text[len(self.emitted):]
                self.emitted = self.text
                return out

        # hold back a tail that could still turn into a stop string
        hold = max(self.max_stop - 1, 0)
        safe = self.text[:len(self.text) - hold] if hold else self.text
        if len(safe) > len(self.emitted):
            out = safe[len(self.emitted):]
            self.emitted = safe
            return out
        return ""

    def finalize(self):
        # Generation can stop mid-codepoint (max_tokens landing between the
        # bytes of an emoji). Batch decode emits U+FFFD there, so we must too,
        # or streaming and non-streaming responses disagree.
        if not self.stopped:
            delta = self._decode_delta(force=True)
            if delta:
                self.text += delta
        out = self.text[len(self.emitted):]
        self.emitted = self.text
        return out
