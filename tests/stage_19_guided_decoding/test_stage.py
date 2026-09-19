"""Stage 19 - constrained output with logit masking.

The spec is in app/s19_guided.py. tests/helpers.py gives you the JSON
validator, because a grammar compiler is a different subject.

This stage is about two things: the alignment of a validator to the
TOKENIZER, and a mask build that stays off the critical path.
"""

import json
import random

import pytest
import torch

from app.s19_guided import (
    GuidedDecoder,
    MaskCache,
    allowed_token_ids,
    apply_logit_mask,
    build_mask,
)
from tests.helpers import json_prefix_state


class FakeTokenizer:
    """A difficult vocabulary on purpose: pieces of several characters, so
    the token boundaries do NOT align with the grammar boundaries. That
    misalignment is the real difficulty of this stage."""

    PIECES = ['{', '}', '[', ']', '"', ':', ',', ' ',
              'a', 'b', 'name', 'age', '"name"', '"age"',
              '1', '2', '10', '42', 'true', 'false', 'null',
              'xyz', '}}', '":"', '<eos>']

    def __init__(self):
        self.pieces = list(self.PIECES)
        self.eos_id = self.pieces.index('<eos>')

    def decode(self, token_ids):
        return "".join(self.pieces[token_id] for token_id in token_ids)

    @property
    def ids(self):
        return [token_id for token_id in range(len(self.pieces))
                if token_id != self.eos_id]

    @property
    def vocab_size(self):
        return len(self.pieces)


@pytest.fixture
def tokenizer():
    return FakeTokenizer()


def allowed_texts(tokenizer, prefix, candidate_ids=None):
    """The text of each token that the validator allows after prefix."""
    allowed = allowed_token_ids(tokenizer, json_prefix_state, prefix,
                                candidate_ids or tokenizer.ids)
    return {tokenizer.decode([token_id]) for token_id in allowed}


def make_decoder(tokenizer):
    return GuidedDecoder(tokenizer, json_prefix_state, tokenizer.ids,
                         eos_id=tokenizer.eos_id,
                         vocab_size=tokenizer.vocab_size)


# ---- the masking primitives -----------------------------------------

def test_build_and_apply_mask():
    masked = apply_logit_mask(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                              build_mask(6, [1, 3]))
    assert masked[1] == 2.0 and masked[3] == 4.0
    assert torch.isinf(masked[0]) and masked[0] < 0
    assert int(masked.argmax()) == 3, "argmax must select an allowed token"


def test_masking_survives_softmax():
    masked = apply_logit_mask(torch.tensor([10.0, 10.0, 0.0, 10.0]),
                              build_mask(4, [2]))
    probs = torch.softmax(masked, dim=-1)
    assert probs[2] == pytest.approx(1.0)
    assert probs[0] == pytest.approx(0.0)


# ---- alignment ------------------------------------------------------

def test_allowed_tokens_at_the_start(tokenizer):
    """Only tokens that can start a JSON value."""
    allowed = allowed_texts(tokenizer, "")
    assert '{' in allowed and '[' in allowed
    assert '}' not in allowed, "'}' cannot start a JSON document"
    assert ':' not in allowed
    assert 'xyz' not in allowed


def test_allowed_tokens_after_an_open_brace(tokenizer):
    allowed = allowed_texts(tokenizer, '{')
    assert '"' in allowed, "a key must start with a quote"
    assert '"name"' in allowed, "a token of several characters that IS a key"
    assert '}' in allowed, "the empty object {} is valid"
    assert '1' not in allowed, "a bare number cannot be an object key"


def test_multi_character_tokens_are_checked_whole(tokenizer):
    """A token is legal only if ALL of its characters are legal.

    '":"' is quote, colon, quote. After '{"name"' only the colon is legal.
    """
    allowed = allowed_texts(tokenizer, '{"name"')
    assert ':' in allowed
    assert '{' not in allowed


def test_completed_value_forbids_continuation(tokenizer):
    allowed = allowed_texts(tokenizer, '{"age":1}')
    assert all(text.strip() == "" for text in allowed), (
        f"nothing can follow a complete document, got {allowed}")


# ---- the guarantee --------------------------------------------------

def _guided_random_walk(decoder, tokenizer, rng, max_steps=40):
    """Random logits through the mask. -> the text."""
    text = ""
    for _ in range(max_steps):
        logits = torch.tensor([rng.gauss(0, 5)
                               for _ in range(tokenizer.vocab_size)])
        masked = apply_logit_mask(logits, decoder.mask_for(text))
        if torch.isinf(masked).all():
            break
        token_id = int(masked.argmax())
        if token_id == tokenizer.eos_id:
            break
        text += tokenizer.decode([token_id])
    return text


def test_constrained_generation_is_always_valid_json(tokenizer):
    """The main check. Random logits, 300 runs, every output parses.

    With no constraint, a random walk over this vocabulary almost never makes
    valid JSON.
    """
    rng = random.Random(0)
    decoder = make_decoder(tokenizer)
    complete = []
    for _ in range(300):
        text = _guided_random_walk(decoder, tokenizer, rng)
        if decoder.is_complete(text):
            json.loads(text)             # raises an error if the mask failed
            complete.append(text)

    assert len(complete) > 30, (
        f"only {len(complete)}/300 runs completed. The mask may be too strict.")
    print(f"\n  {len(complete)}/300 runs made complete JSON, and all of it "
          "parses")
    print(f"  samples: {complete[:4]}")


def test_unconstrained_generation_is_essentially_never_valid(tokenizer):
    """This checks none of your code. It is the baseline that justifies the
    stage."""
    rng = random.Random(0)
    num_valid = 0
    for _ in range(300):
        text = "".join(tokenizer.decode([rng.choice(tokenizer.ids)])
                       for _ in range(6))
        try:
            json.loads(text)
            num_valid += 1
        except ValueError:
            pass
    print(f"\n  unconstrained: {num_valid}/300 were valid JSON by chance")
    assert num_valid < 15


# ---- off the critical path ------------------------------------------

def test_mask_cache_reports_hits_and_misses():
    built_keys = []

    def builder(key):
        built_keys.append(key)
        return build_mask(4, [0])

    cache = MaskCache(builder)
    cache.get("a")
    cache.get("a")
    cache.get("b")
    assert cache.misses == 2 and cache.hits == 1
    assert len(built_keys) == 2, "a cache hit must not build the mask again"


def test_cache_makes_repeated_states_cheap(tokenizer):
    """A mask build is O(vocab) validator calls. It must not run at each step.

    Real implementations key on the grammar STATE, compute masks before
    they need them, and build masks during the forward pass.
    """
    decoder = make_decoder(tokenizer)
    for _ in range(50):
        decoder.mask_for('{"name"')
    assert decoder.cache.misses == 1
    assert decoder.cache.hits == 49
    print(f"\n  50 lookups of the same state -> {decoder.cache.misses} build, "
          f"{decoder.cache.hits} hits")


def test_works_with_a_real_tokenizer(hf):
    """BPE pieces, a real vocabulary. A small candidate set, so that it stays
    fast."""
    _, real_tokenizer = hf
    candidate_ids = []
    for piece in ['{', '}', '"', ':', ',', '[', ']', 'name', 'age', '1', '2',
                  'true', 'null', ' ']:
        piece_ids = real_tokenizer(piece, add_special_tokens=False).input_ids
        if len(piece_ids) == 1:
            candidate_ids.append(piece_ids[0])
    assert len(candidate_ids) > 6, "expected several single-token JSON pieces"

    allowed = allowed_texts(real_tokenizer, "", candidate_ids)
    assert '{' in allowed
    assert '}' not in allowed
    print(f"\n  real tokenizer, allowed at the start: {sorted(allowed)}")
