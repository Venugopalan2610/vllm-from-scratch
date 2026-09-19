"""Reference solution, stage 19 - guided decoding with a logit mask."""

import torch

LEGAL_STATES = ("valid", "prefix")


def allowed_token_ids(tokenizer, validator, prefix, candidate_ids):
    """The tokens whose text keeps the prefix 'valid' or 'prefix'."""
    return [token_id for token_id in candidate_ids
            if validator(prefix + tokenizer.decode([token_id])) in LEGAL_STATES]


def build_mask(vocab_size, allowed_ids, device="cpu"):
    mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    if allowed_ids:
        mask[torch.tensor(list(allowed_ids), dtype=torch.long,
                          device=device)] = True
    return mask


def apply_logit_mask(logits, mask):
    """True in the mask = allowed. A position that is not allowed becomes
    -inf."""
    return logits.masked_fill(~mask, float("-inf"))


class MaskCache:
    """Keeps each mask under a key of the grammar state.

    The key makes this possible. A key of the full text gives a hit rate of
    about 0%. A key of the grammar STATE lets a few masks cover a whole
    generation.
    """

    def __init__(self, builder):
        self.builder = builder
        self.masks = {}
        self.hits = 0
        self.misses = 0

    def get(self, key):
        if key in self.masks:
            self.hits += 1
            return self.masks[key]
        self.misses += 1
        self.masks[key] = self.builder(key)
        return self.masks[key]

    @property
    def size(self):
        return len(self.masks)


class GuidedDecoder:
    def __init__(self, tokenizer, validator, candidate_ids, eos_id=None,
                 vocab_size=None, device="cpu"):
        self.tokenizer = tokenizer
        self.validator = validator
        self.candidate_ids = list(candidate_ids)
        self.eos_id = eos_id
        self.vocab_size = vocab_size or (max(self.candidate_ids) + 2)
        self.device = device
        self.cache = MaskCache(self._build)

    def _build(self, key):
        prefix, state = key
        allowed = allowed_token_ids(self.tokenizer, self.validator, prefix,
                                    self.candidate_ids)
        if self.eos_id is not None and state == "valid":
            allowed.append(self.eos_id)
        return build_mask(self.vocab_size, allowed, self.device)

    def state(self, text):
        return self.validator(text)

    def mask_for(self, text):
        return self.cache.get((text, self.state(text)))

    def is_complete(self, text):
        return self.state(text) == "valid"
