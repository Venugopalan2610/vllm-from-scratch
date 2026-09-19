"""Stage 26 - guided JSON decoding inside the engine.

`./vc lore 26` for the insight. `./vc test 26` to check yourself.

Stage 19 built a mask from a validator that parses the whole text again for
every candidate token. For a vocabulary of 150,000 tokens that costs seconds
for each step. Its cache also keys on the full text, so it never hits. It
proved the idea. It cannot serve.

This stage builds what XGrammar and Outlines build: a character automaton for
JSON, and masks cached by AUTOMATON STATE. A whole JSON answer needs a few
dozen states, and the cache keeps them across requests.

WHAT YOU ARE BUILDING

    Mode, JsonState, START, NUMBER_NEXT     given: the states and the numbers
    step(state, char) -> the next state, or None if char is illegal here
    run(state, text) -> the state after text, or None
    distance(state) -> the fewest characters that complete the JSON
    TokenTable(tokenizer, vocab_size)       the text of every token, one time
    mask_key(state, forced) -> the cache key of a mask
    allowed_tokens(table, state, forced, eos_ids) -> legal token ids
    JsonMasker(tokenizer, vocab_size, device, eos_ids)
        .mask(state, forced=False) -> (vocab_size,) bool on the device
        .advance(state, token_id) -> the next state
        .cache, .hits, .misses
    GuidedEngine(*args, **kwargs)           your stage 25 engine, and:
        .add_request(..., json_mode=False)
        _propose, _mask and _on_token overridden
    is_valid_json(text) -> bool

THE AUTOMATON

A JsonState is (stack, mode, detail). The stack holds 'object' and 'array'
for the open containers. The class Mode names every mode, and the docstring
of JsonState says what detail holds in each one. One small function handles
each mode. JSON mode asks for an OBJECT at the top level, as the OpenAI API
does.

A token is legal when the automaton accepts every character of its text.

MAKE A MASK BUILD CHEAP

  - Decode every token one time, at startup. Skip the special tokens: their
    text is a name, not output.
  - Inside a string, a token with no quote, no backslash and no control
    character leaves the state as it was. That is most of the vocabulary.
    Allow all of those tokens with no simulation.
  - Outside a string, a token is legal only if its FIRST character is. Group
    the tokens by first character, and test each group once.
  - Key the cache on the last few entries of the stack, not the whole stack.
    A token that pops below that part cannot be checked. Reject it. The mask
    can be stricter than the grammar. It must never admit an illegal token.

FINISH THE OBJECT

Part 7 showed that one fifth of the masked runs reach max_tokens with open
brackets. When the tokens left are at most distance(state) + 2, admit only
tokens that make distance() smaller. In DONE, admit only the stop tokens.

TRAPS

  - A guided request does not speculate: a draft skips the mask.
  - Build the masks on the GPU, as bool tensors, one time. masked_fill on the
    row is then one kernel.
"""

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
    raise NotImplementedError("stage 26: implement _start_value")


def _close(stack, char):
    raise NotImplementedError("stage 26: implement _close")


def _in_string(state, char):
    raise NotImplementedError("stage 26: implement _in_string")


def _in_escape(state, char):
    raise NotImplementedError("stage 26: implement _in_escape")


def _in_unicode(state, char):
    raise NotImplementedError("stage 26: implement _in_unicode")


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
    raise NotImplementedError("stage 26: implement _in_number")


def _in_literal(state, char):
    raise NotImplementedError("stage 26: implement _in_literal")


def _structural(state, char):
    """The modes outside strings, numbers and literals."""
    raise NotImplementedError("stage 26: implement _structural")


MODE_STEPS = {Mode.STRING: _in_string, Mode.ESCAPE: _in_escape,
              Mode.UNICODE: _in_unicode, Mode.NUMBER: _in_number,
              Mode.LITERAL: _in_literal}


def step(state, char):
    """The state after one character, or None if the character is illegal."""
    raise NotImplementedError("stage 26: implement step")


def run(state, text):
    raise NotImplementedError("stage 26: implement run")


def _string_distance(state):
    """Finish the escape, close the quote, and for a key add ':0'."""
    raise NotImplementedError("stage 26: implement _string_distance")


def distance(state):
    """The fewest characters that complete the JSON from this state."""
    raise NotImplementedError("stage 26: implement distance")


# ---------------------------------------------------------------- tokens


class TokenTable:
    """The text of every token, decoded one time."""

    def __init__(self, tokenizer, vocab_size):
        raise NotImplementedError("stage 26: implement TokenTable.__init__")


def _changes_a_string(text):
    raise NotImplementedError("stage 26: implement _changes_a_string")


# ---------------------------------------------------------------- masks


STACK_IN_KEY = 6    # the stack entries that a cache key keeps


def mask_key(state, forced):
    """A mask depends on the state, and only on the top of the stack."""
    raise NotImplementedError("stage 26: implement mask_key")


def _token_is_legal(table, state, token_id, forced, stack_is_cut):
    raise NotImplementedError("stage 26: implement _token_is_legal")


def allowed_tokens(table, state, forced, eos_ids):
    """The token ids that the automaton accepts in this state. forced: only
    tokens that bring the JSON closer to complete."""
    raise NotImplementedError("stage 26: implement allowed_tokens")


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
        raise NotImplementedError("stage 26: implement JsonMasker._build")

    def mask(self, state, forced=False):
        raise NotImplementedError("stage 26: implement JsonMasker.mask")

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
        raise NotImplementedError("stage 26: implement GuidedEngine._propose")

    def _mask(self, seq, logits):
        raise NotImplementedError("stage 26: implement GuidedEngine._mask")

    def _on_token(self, seq, token):
        raise NotImplementedError("stage 26: implement GuidedEngine._on_token")


def is_valid_json(text):
    try:
        json.loads(text)
        return True
    except ValueError:
        return False
