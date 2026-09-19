"""Reference solution, stage 14 - incremental detokenization."""

REPLACEMENT_CHARACTER = "�"


class IncrementalDetokenizer:
    def __init__(self, tokenizer, stop_strings=()):
        self.tokenizer = tokenizer
        self.stop_strings = list(stop_strings)
        self.longest_stop = max((len(s) for s in self.stop_strings), default=0)
        self.tokens = []
        self.prefix_offset = 0
        self.read_offset = 0
        self.text = ""        # all decoded text, cut at a stop string
        self.emitted = ""     # all text given to the caller
        self.stopped = False
        self.stop_reason = None

    def _decode(self, start, end=None):
        return self.tokenizer.decode(self.tokens[start:end],
                                     skip_special_tokens=False)

    def _decode_delta(self, force=False):
        """The prefix offset and read offset method of vLLM.

        Decode two windows that overlap, and compare them. If the new window
        ends in a replacement character, a multi-byte character is not
        complete. Then return nothing, and wait for the next token.
        """
        prefix_text = self._decode(self.prefix_offset, self.read_offset)
        new_text = self._decode(self.prefix_offset)
        if len(new_text) <= len(prefix_text):
            return ""
        if new_text.endswith(REPLACEMENT_CHARACTER) and not force:
            return ""
        self.prefix_offset = self.read_offset
        self.read_offset = len(self.tokens)
        return new_text[len(prefix_text):]

    def _emit_up_to(self, end):
        """Give the caller the text from the last emit up to `end`."""
        if end <= len(self.emitted):
            return ""
        delta = self.text[len(self.emitted):end]
        self.emitted = self.text[:end]
        return delta

    def _cut_at_stop(self):
        """-> True if the text holds a stop string. The text ends before it."""
        for stop in self.stop_strings:
            index = self.text.find(stop)
            if index != -1:
                self.stopped = True
                self.stop_reason = stop
                self.text = self.text[:index]
                return True
        return False

    def add_token(self, token_id):
        if self.stopped:
            return ""
        self.tokens.append(token_id)
        delta = self._decode_delta()
        if not delta:
            return ""
        self.text += delta
        if self._cut_at_stop():
            return self._emit_up_to(len(self.text))
        # Keep back a tail that can still become a stop string.
        held_back = max(self.longest_stop - 1, 0)
        return self._emit_up_to(len(self.text) - held_back)

    def finalize(self):
        # A generation can stop inside a character (max_tokens between the
        # bytes of an emoji). A batch decode gives U+FFFD there, so this must
        # too. If not, a stream and a whole response do not agree.
        if not self.stopped:
            self.text += self._decode_delta(force=True)
        return self._emit_up_to(len(self.text))
