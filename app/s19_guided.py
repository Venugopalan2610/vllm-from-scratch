"""Stage 19 - constrained output via logit masking.

`./vc lore 19` for the insight. `./vc test 19` to check yourself.

"Respond in JSON" in a prompt is a request. Logit masking is a guarantee:
at every step, compute the tokens that can legally come next, and set every
other logit to -inf. The model then cannot write illegal output.

tests/helpers.py gives you the grammar validator. To compile a schema to an
automaton is a different subject. Two other parts are important here, and
they are what breaks in production:

  1. TOKENIZER ALIGNMENT. The grammar works in characters. The model works
     in BPE tokens, and the two do not align. One token can hold a closing
     quote, a comma and the start of the next key. A token is legal only if
     EVERY character in it is legal.

  2. KEEP IT OFF THE CRITICAL PATH. To make a mask, you ask the validator
     about every candidate token. Do that inside the decode loop, and you
     add milliseconds to a step of a few milliseconds. Cache by grammar
     STATE, and make masks while the GPU runs the forward pass.
"""

import torch


def allowed_token_ids(tokenizer, validator, prefix, candidate_ids):
    """The candidates that keep the output on a path to valid output.

    Allow a token if validator(prefix + tokenizer.decode([token_id])) is
    'valid' or 'prefix'. Refuse only 'invalid'. 'prefix' means "not
    complete, but it can still become valid", and that is most of a
    generation.
    """
    raise NotImplementedError("stage 19: implement allowed_token_ids")


def build_mask(vocab_size, allowed_ids, device="cpu"):
    """A bool tensor over the vocabulary: True = allowed."""
    raise NotImplementedError("stage 19: implement build_mask")


def apply_logit_mask(logits, mask):
    """A position that is not allowed -> -inf, so softmax gives it exactly
    zero.

    Use -inf, not a large negative number. A finite penalty leaves a small
    probability that a long generation samples one day. Then your
    "guaranteed" JSON is not valid one time in a few thousand requests.
    """
    raise NotImplementedError("stage 19: implement apply_logit_mask")


class MaskCache:
    """Keep the masks by key. Required: .hits .misses .size, get(key).

    The key is the most important decision. A key of the full text gives a
    hit rate of about 0%, because the text changes at every step. A key of
    the grammar STATE lets a few masks cover a whole generation.
    """

    def __init__(self, builder):
        raise NotImplementedError("stage 19: implement MaskCache")


class GuidedDecoder:
    """Required:
        .cache
        state(text) -> str
        mask_for(text) -> BoolTensor
        is_complete(text) -> bool

    If the caller gives eos_id, allow it ONLY when the text is already
    'valid'. That stops the model from finishing in the middle of an object.
    """

    def __init__(self, tokenizer, validator, candidate_ids, eos_id=None,
                 vocab_size=None, device="cpu"):
        raise NotImplementedError("stage 19: implement GuidedDecoder")
