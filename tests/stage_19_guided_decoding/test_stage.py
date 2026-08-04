"""Stage 19 - Constrained output via logit masking.

Spec in app/s19_guided.py. The JSON validator is supplied in tests/helpers.py --
compiling a grammar is a separate discipline. This stage is about aligning a
validator to the TOKENIZER and keeping mask construction off the critical path.
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
    """A deliberately awkward vocabulary: multi-character pieces, so token
    boundaries do NOT line up with grammar boundaries. That misalignment is
    the actual difficulty of this stage."""

    PIECES = ['{', '}', '[', ']', '"', ':', ',', ' ',
              'a', 'b', 'name', 'age', '"name"', '"age"',
              '1', '2', '10', '42', 'true', 'false', 'null',
              'xyz', '}}', '":"', '<eos>']

    def __init__(self):
        self.pieces = list(self.PIECES)
        self.eos_id = self.pieces.index('<eos>')

    def decode(self, ids):
        return "".join(self.pieces[i] for i in ids)

    @property
    def ids(self):
        return [i for i in range(len(self.pieces)) if i != self.eos_id]

    @property
    def vocab_size(self):
        return len(self.pieces)


@pytest.fixture
def tk():
    return FakeTokenizer()


# ---- masking primitives ---------------------------------------------

def test_build_and_apply_mask():
    mask = build_mask(6, [1, 3])
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    out = apply_logit_mask(logits, mask)
    assert out[1] == 2.0 and out[3] == 4.0
    assert torch.isinf(out[0]) and out[0] < 0
    assert int(out.argmax()) == 3, "argmax must land on an allowed token"


def test_masking_survives_softmax():
    mask = build_mask(4, [2])
    logits = torch.tensor([10.0, 10.0, 0.0, 10.0])
    probs = torch.softmax(apply_logit_mask(logits, mask), dim=-1)
    assert probs[2] == pytest.approx(1.0)
    assert probs[0] == pytest.approx(0.0)


# ---- alignment ------------------------------------------------------

def test_allowed_tokens_at_the_start(tk):
    """Only tokens that could begin a JSON value."""
    ids = allowed_token_ids(tk, json_prefix_state, "", tk.ids)
    got = {tk.decode([i]) for i in ids}
    assert '{' in got and '[' in got
    assert '}' not in got, "'}' cannot start a JSON document"
    assert ':' not in got
    assert 'xyz' not in got


def test_allowed_tokens_after_an_open_brace(tk):
    ids = allowed_token_ids(tk, json_prefix_state, '{', tk.ids)
    got = {tk.decode([i]) for i in ids}
    assert '"' in got, 'a key must start with a quote'
    assert '"name"' in got, "a multi-character token that IS a whole key"
    assert '}' in got, "the empty object {} is valid"
    assert '1' not in got, "a bare number cannot be an object key"


def test_multi_character_tokens_are_checked_whole(tk):
    """A token is legal only if ALL of its characters are.

    '":"' is quote-colon-quote. After '{' it would open a key and immediately
    close it and start a colon, which is not valid there.
    """
    ids = allowed_token_ids(tk, json_prefix_state, '{"name"', tk.ids)
    got = {tk.decode([i]) for i in ids}
    assert ':' in got
    assert '{' not in got


def test_completed_value_forbids_continuation(tk):
    ids = allowed_token_ids(tk, json_prefix_state, '{"age":1}', tk.ids)
    got = {tk.decode([i]) for i in ids}
    assert got == set() or all(g.strip() == "" for g in got), (
        f"nothing may follow a complete document, got {got}"
    )


# ---- the guarantee --------------------------------------------------

def test_constrained_generation_is_always_valid_json(tk):
    """The headline. Random logits, 300 runs, every output parses.

    Unconstrained, a random walk over this vocabulary produces valid JSON
    essentially never.
    """
    rng = random.Random(0)
    gd = GuidedDecoder(tk, json_prefix_state, tk.ids, eos_id=tk.eos_id,
                       vocab_size=tk.vocab_size)

    produced = []
    for trial in range(300):
        text = ""
        for _ in range(40):
            logits = torch.tensor([rng.gauss(0, 5) for _ in range(tk.vocab_size)])
            masked = apply_logit_mask(logits, gd.mask_for(text))
            if torch.isinf(masked).all():
                break
            tid = int(masked.argmax())
            if tid == tk.eos_id:
                break
            text += tk.decode([tid])
        if gd.is_complete(text):
            produced.append(text)
            json.loads(text)          # raises if we lied

    assert len(produced) > 30, (
        f"only {len(produced)}/300 runs completed -- the mask may be too tight"
    )
    print(f"\n  {len(produced)}/300 runs produced complete JSON, all parseable")
    print(f"  samples: {produced[:4]}")


def test_unconstrained_generation_is_essentially_never_valid(tk):
    """Not a test of your code -- the baseline that justifies the stage."""
    rng = random.Random(0)
    ok = 0
    for _ in range(300):
        text = "".join(tk.decode([rng.choice(tk.ids)]) for _ in range(6))
        try:
            json.loads(text)
            ok += 1
        except Exception:
            pass
    print(f"\n  unconstrained: {ok}/300 happened to be valid JSON")
    assert ok < 15


# ---- keeping it off the critical path -------------------------------

def test_mask_cache_reports_hits_and_misses():
    calls = []

    def builder(key):
        calls.append(key)
        return build_mask(4, [0])

    c = MaskCache(builder)
    c.get("a")
    c.get("a")
    c.get("b")
    assert c.misses == 2 and c.hits == 1
    assert len(calls) == 2, "a cache hit must not rebuild the mask"


def test_cache_makes_repeated_states_cheap(tk):
    """Mask construction is O(vocab) validator calls. It must not run per step.

    Real implementations key on the grammar STATE, precompute masks when the
    request arrives, and overlap construction with the forward pass.
    """
    gd = GuidedDecoder(tk, json_prefix_state, tk.ids, eos_id=tk.eos_id,
                       vocab_size=tk.vocab_size)
    for _ in range(50):
        gd.mask_for('{"name"')
    assert gd.cache.misses == 1
    assert gd.cache.hits == 49
    print(f"\n  50 lookups of the same state -> {gd.cache.misses} build, "
          f"{gd.cache.hits} hits")


def test_works_with_a_real_tokenizer(hf):
    """BPE pieces, real vocabulary. Restricted candidate set to stay fast."""
    _, tok = hf
    cands = []
    for s in ['{', '}', '"', ':', ',', '[', ']', 'name', 'age', '1', '2',
              'true', 'null', ' ']:
        ids = tok(s, add_special_tokens=False).input_ids
        if len(ids) == 1:
            cands.append(ids[0])
    assert len(cands) > 6, "expected several single-token JSON pieces"

    allowed = allowed_token_ids(tok, json_prefix_state, "", cands)
    got = {tok.decode([i]) for i in allowed}
    assert '{' in got
    assert '}' not in got
    print(f"\n  real tokenizer, allowed at start: {sorted(got)}")
