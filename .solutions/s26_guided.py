"""Reference solution, stage 26 - guided JSON decoding inside the engine."""

import json
from typing import NamedTuple

import torch

from app.s25_speculative import SpeculativeEngine

WHITESPACE = " \t\n\r"
HEX_DIGITS = "0123456789abcdefABCDEF"


class Mode:
    START = "start"                        # before the top-level object
    VALUE = "value"                        # a value must come next
    VALUE_OR_ARRAY_END = "value_or_]"      # just after '['
    KEY_OR_OBJECT_END = "key_or_}"         # just after '{'
    KEY = "key"                            # just after ',' in an object
    STRING = "string"
    ESCAPE = "escape"                      # just after a backslash
    UNICODE = "unicode"                    # in \uXXXX
    COLON = "colon"                        # just after a key
    AFTER_VALUE = "after_value"            # ',' or a closer
    NUMBER = "number"
    LITERAL = "literal"                    # in true, false or null
    DONE = "done"                          # the top-level object is complete


class JsonState(NamedTuple):
    """stack holds 'object' and 'array' for the open containers. detail is
    the part of the state that one mode needs:
      STRING, ESCAPE     'key' or 'value'
      UNICODE            ('key' or 'value', hex digits still due)
      NUMBER             the part of the number that is next
      LITERAL            the letters still due"""
    stack: tuple
    mode: str
    detail: object = None


START = JsonState((), Mode.START)


# ---------------------------------------------------------------- automaton


def _after_a_value(stack):
    if not stack:
        return JsonState(stack, Mode.DONE)
    return JsonState(stack, Mode.AFTER_VALUE)


def _start_value(stack, char):
    if char == '"':
        return JsonState(stack, Mode.STRING, "value")
    if char == "{":
        return JsonState(stack + ("object",), Mode.KEY_OR_OBJECT_END)
    if char == "[":
        return JsonState(stack + ("array",), Mode.VALUE_OR_ARRAY_END)
    if char == "-":
        return JsonState(stack, Mode.NUMBER, "after_minus")
    if char == "0":
        return JsonState(stack, Mode.NUMBER, "after_zero")
    if char in "123456789":
        return JsonState(stack, Mode.NUMBER, "integer")
    for literal in ("true", "false", "null"):
        if char == literal[0]:
            return JsonState(stack, Mode.LITERAL, literal[1:])
    return None


def _close(stack, char):
    if char == "}" and stack and stack[-1] == "object":
        return _after_a_value(stack[:-1])
    if char == "]" and stack and stack[-1] == "array":
        return _after_a_value(stack[:-1])
    return None


def _in_string(state, char):
    if char == '"':
        if state.detail == "key":
            return JsonState(state.stack, Mode.COLON)
        return _after_a_value(state.stack)
    if char == "\\":
        return JsonState(state.stack, Mode.ESCAPE, state.detail)
    return None if ord(char) < 0x20 else state


def _in_escape(state, char):
    if char in '"\\/bfnrt':
        return JsonState(state.stack, Mode.STRING, state.detail)
    if char == "u":
        return JsonState(state.stack, Mode.UNICODE, (state.detail, 4))
    return None


def _in_unicode(state, char):
    if char not in HEX_DIGITS:
        return None
    kind, digits_left = state.detail
    if digits_left == 1:
        return JsonState(state.stack, Mode.STRING, kind)
    return JsonState(state.stack, Mode.UNICODE, (kind, digits_left - 1))


# The number grammar: for each part, the part after each class of character.
# A number ends at the first other character, if its part may end.
NUMBER_NEXT = {
    "after_minus": {"0": "after_zero", "1-9": "integer"},
    "after_zero": {".": "after_dot", "e": "after_e"},
    "integer": {"0-9": "integer", ".": "after_dot", "e": "after_e"},
    "after_dot": {"0-9": "fraction"},
    "fraction": {"0-9": "fraction", "e": "after_e"},
    "after_e": {"+-": "after_exponent_sign", "0-9": "exponent"},
    "after_exponent_sign": {"0-9": "exponent"},
    "exponent": {"0-9": "exponent"},
}
NUMBER_CAN_END = {"after_zero", "integer", "fraction", "exponent"}
CHARACTER_CLASSES = {"0": "0", "1-9": "123456789", "0-9": "0123456789",
                     ".": ".", "e": "eE", "+-": "+-"}


def _in_number(state, char):
    for char_class, next_part in NUMBER_NEXT[state.detail].items():
        if char in CHARACTER_CLASSES[char_class]:
            return JsonState(state.stack, Mode.NUMBER, next_part)
    if state.detail not in NUMBER_CAN_END:
        return None
    return step(_after_a_value(state.stack), char)      # the number ended


def _in_literal(state, char):
    if char != state.detail[0]:
        return None
    if len(state.detail) == 1:
        return _after_a_value(state.stack)
    return JsonState(state.stack, Mode.LITERAL, state.detail[1:])


def _structural(state, char):
    """The modes outside strings, numbers and literals."""
    stack = state.stack
    if char in WHITESPACE:
        return state
    if state.mode == Mode.START:
        return JsonState(("object",), Mode.KEY_OR_OBJECT_END) if char == "{" else None
    if state.mode == Mode.VALUE:
        return _start_value(stack, char)
    if state.mode == Mode.VALUE_OR_ARRAY_END:
        return _close(stack, char) if char == "]" else _start_value(stack, char)
    if state.mode == Mode.KEY_OR_OBJECT_END:
        return JsonState(stack, Mode.STRING, "key") if char == '"' else _close(stack, char)
    if state.mode == Mode.KEY:
        return JsonState(stack, Mode.STRING, "key") if char == '"' else None
    if state.mode == Mode.COLON:
        return JsonState(stack, Mode.VALUE) if char == ":" else None
    if state.mode == Mode.AFTER_VALUE and char == ",":
        return JsonState(stack, Mode.KEY if stack[-1] == "object" else Mode.VALUE)
    if state.mode == Mode.AFTER_VALUE:
        return _close(stack, char)
    return None                                           # DONE takes nothing


MODE_STEPS = {Mode.STRING: _in_string, Mode.ESCAPE: _in_escape,
              Mode.UNICODE: _in_unicode, Mode.NUMBER: _in_number,
              Mode.LITERAL: _in_literal}


def step(state, char):
    """The state after one character, or None if the character is illegal."""
    return MODE_STEPS.get(state.mode, _structural)(state, char)


def run(state, text):
    for char in text:
        state = step(state, char)
        if state is None:
            return None
    return state


def _string_distance(state):
    """Finish the escape, close the quote, and for a key add ':0'."""
    if state.mode == Mode.UNICODE:
        kind, escape_left = state.detail
    else:
        kind = state.detail
        escape_left = 1 if state.mode == Mode.ESCAPE else 0
    key_tail = 2 if kind == "key" else 0
    return escape_left + 1 + key_tail


def distance(state):
    """The fewest characters that complete the JSON from this state."""
    closers = len(state.stack)
    mode = state.mode
    if mode == Mode.DONE:
        return 0
    if mode == Mode.START:
        return 2                                          # {}
    if mode in (Mode.STRING, Mode.ESCAPE, Mode.UNICODE):
        return _string_distance(state) + closers
    if mode == Mode.COLON:
        return 2 + closers                                # :0
    if mode == Mode.VALUE:
        return 1 + closers                                # 0
    if mode == Mode.KEY:
        return 4 + closers                                # "":0
    if mode == Mode.NUMBER:
        return (0 if state.detail in NUMBER_CAN_END else 1) + closers
    if mode == Mode.LITERAL:
        return len(state.detail) + closers
    return closers                                        # the closers only


# ---------------------------------------------------------------- tokens


class TokenTable:
    """The text of every token, decoded one time."""

    def __init__(self, tokenizer, vocab_size):
        special = set(tokenizer.all_special_ids)
        special |= set(getattr(tokenizer, "added_tokens_decoder", {}))
        self.text = {}
        for token_id in range(min(len(tokenizer), vocab_size)):
            if token_id not in special:           # a special name is not output
                text = tokenizer.decode([token_id])
                if text:
                    self.text[token_id] = text
        # In a string, a token with no quote, no backslash and no control
        # character leaves the state unchanged.
        self.plain_in_string = [t for t, text in self.text.items()
                                if not _changes_a_string(text)]
        plain = set(self.plain_in_string)
        self.special_in_string = [t for t in self.text if t not in plain]
        self.by_first_char = {}
        for token_id, text in self.text.items():
            self.by_first_char.setdefault(text[0], []).append(token_id)


def _changes_a_string(text):
    return '"' in text or "\\" in text or any(ord(c) < 0x20 for c in text)


# ---------------------------------------------------------------- masks


STACK_IN_KEY = 6    # the stack entries that a cache key keeps


def mask_key(state, forced):
    """A mask depends on the state, and only on the top of the stack."""
    return (state.stack[-STACK_IN_KEY:], len(state.stack) > STACK_IN_KEY,
            state.mode, state.detail, forced)


def _token_is_legal(table, state, token_id, forced, stack_is_cut):
    after = run(state, table.text[token_id])
    if after is None:
        return False
    # The key keeps only the top of the stack. A token that pops below it
    # needs entries that the key does not have. Reject it: the mask may be
    # strict, never wrong.
    if stack_is_cut and (not after.stack or after.mode == Mode.DONE):
        return False
    return not forced or distance(after) < distance(state)


def allowed_tokens(table, state, forced, eos_ids):
    """The token ids that the automaton accepts in this state. forced: only
    tokens that bring the JSON closer to complete."""
    if state.mode == Mode.DONE:
        return list(eos_ids)
    stack_is_cut = len(state.stack) > STACK_IN_KEY
    probe = JsonState(state.stack[-STACK_IN_KEY:], state.mode, state.detail)

    def legal(token_id):
        return _token_is_legal(table, probe, token_id, forced, stack_is_cut)

    if state.mode == Mode.STRING and not forced:
        return table.plain_in_string + [t for t in table.special_in_string
                                        if legal(t)]
    allowed = []
    for first_char, token_ids in table.by_first_char.items():
        if step(probe, first_char) is not None:
            allowed += [t for t in token_ids if legal(t)]
    return allowed


class JsonMasker:
    """Token masks for the JSON automaton, cached by automaton state."""

    def __init__(self, tokenizer, vocab_size, device, eos_ids):
        self.table = TokenTable(tokenizer, vocab_size)
        self.vocab_size = vocab_size
        self.device = device
        self.eos_ids = sorted(eos_ids)
        self.cache = {}
        self.hits = 0
        self.misses = 0

    def _build(self, state, forced):
        mask = torch.zeros(self.vocab_size, dtype=torch.bool, device=self.device)
        token_ids = allowed_tokens(self.table, state, forced, self.eos_ids)
        if token_ids:
            mask[torch.tensor(token_ids, device=self.device)] = True
        return mask

    def mask(self, state, forced=False):
        key = mask_key(state, forced)
        if key in self.cache:
            self.hits += 1
        else:
            self.misses += 1
            self.cache[key] = self._build(state, forced)
        return self.cache[key]

    def advance(self, state, token_id):
        if token_id in self.eos_ids:
            return state
        return run(state, self.table.text[token_id])


# ---------------------------------------------------------------- engine


class GuidedEngine(SpeculativeEngine):
    """The stage 25 engine, with a JSON mode for each request."""

    FORCE_MARGIN = 2      # start to close the object this many tokens early

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.masker = JsonMasker(self.tokenizer, self.model.config.vocab_size,
                                 self.runner.device, self.model.eos_ids)
        self.json_states = {}

    def add_request(self, rid, prompt, max_tokens=16, params=None, stop=(),
                    ignore_eos=False, arrival=None, json_mode=False):
        super().add_request(rid, prompt, max_tokens, params, stop, ignore_eos,
                            arrival)
        if json_mode:
            self.json_states[rid] = START

    def _propose(self, seq, room):
        if seq.rid in self.json_states:
            return []                         # a draft would skip the mask
        return super()._propose(seq, room)

    def _mask(self, seq, logits):
        state = self.json_states.get(seq.rid)
        if state is None:
            return logits
        forced = seq.tokens_left <= distance(state) + self.FORCE_MARGIN
        return logits.masked_fill(~self.masker.mask(state, forced),
                                  float("-inf"))

    def _on_token(self, seq, token):
        if seq.rid in self.json_states:
            self.json_states[seq.rid] = self.masker.advance(
                self.json_states[seq.rid], token)


def is_valid_json(text):
    try:
        json.loads(text)
        return True
    except ValueError:
        return False
