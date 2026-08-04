"""Stage 19 - constrained output via logit masking.

`./vc lore 19` for the insight. `./vc test 19` to check yourself.

"Respond in JSON" as a prompt instruction is a request. Logit masking is a
guarantee: at every step, compute which tokens could legally come next and set
every other logit to -inf. Illegal output becomes unrepresentable.

The grammar validator is supplied for you in tests/helpers.py -- compiling a
schema to an automaton is its own discipline. What matters here, and what
actually breaks in production, is the other two parts:

  1. TOKENIZER ALIGNMENT. The grammar thinks in characters; the model thinks in
     BPE tokens, and they do not line up. A single token can span a closing
     quote, a comma, and the start of the next key. A token is legal only if
     EVERY character in it is legal.

  2. KEEPING IT OFF THE CRITICAL PATH. Building a mask means asking the
     validator about every candidate token. Do that inside the decode loop and
     you have added milliseconds to a step that takes milliseconds. Cache by
     grammar STATE, and build masks while the GPU is busy with the forward pass.
"""

import torch


def allowed_token_ids(tokenizer, validator, prefix, candidate_ids):
    """Which candidates keep us on a path to valid output.

    A token is allowed if validator(prefix + tokenizer.decode([tid])) is
    'valid' or 'prefix'. Only 'invalid' is excluded -- 'prefix' means
    incomplete-but-still-rescuable, which is most of generation.
    """
    raise NotImplementedError("stage 19: implement allowed_token_ids")


def build_mask(vocab_size, allowed_ids, device="cpu"):
    """Bool tensor over the vocabulary: True = allowed."""
    raise NotImplementedError("stage 19: implement build_mask")


def apply_logit_mask(logits, mask):
    """Disallowed positions -> -inf, so softmax gives them exactly zero.

    Use -inf, not a large negative number. A finite penalty leaves a small
    probability that a long generation will eventually sample, and then your
    "guaranteed" JSON is invalid once every few thousand requests.
    """
    raise NotImplementedError("stage 19: implement apply_logit_mask")


class MaskCache:
    """Memoize masks by key. Required: .hits .misses .size, get(key).

    Key choice is the whole game. Key on full text and you get a ~0% hit rate,
    because the text is different every step. Key on the grammar STATE and a
    handful of masks cover an entire generation.
    """

    def __init__(self, builder):
        raise NotImplementedError("stage 19: implement MaskCache")


class GuidedDecoder:
    """Required:
        .cache
        state(text) -> str
        mask_for(text) -> BoolTensor
        is_complete(text) -> bool

    If eos_id is given, it is allowed ONLY when the current text is already
    'valid' -- that is what stops the model finishing mid-object.
    """

    def __init__(self, tokenizer, validator, candidate_ids, eos_id=None,
                 vocab_size=None, device="cpu"):
        raise NotImplementedError("stage 19: implement GuidedDecoder")
