"""Reference solution, stage 19 - guided decoding via logit masking."""

import torch


def allowed_token_ids(tokenizer, validator, prefix, candidate_ids):
    """Tokens whose text keeps prefix in {'valid', 'prefix'}."""
    out = []
    for tid in candidate_ids:
        piece = tokenizer.decode([tid])
        if validator(prefix + piece) in ("valid", "prefix"):
            out.append(tid)
    return out


def build_mask(vocab_size, allowed_ids, device="cpu"):
    m = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    if allowed_ids:
        m[torch.tensor(list(allowed_ids), dtype=torch.long, device=device)] = True
    return m


def apply_logit_mask(logits, mask):
    """mask True = allowed. Disallowed positions become -inf."""
    return logits.masked_fill(~mask, float("-inf"))


class MaskCache:
    """Memoizes masks by grammar-state key.

    The key is what makes this viable. Keying on the full text gives you a
    ~0% hit rate; keying on the grammar STATE means a handful of distinct
    masks cover an entire generation.
    """

    def __init__(self, builder):
        self.builder = builder
        self._cache = {}
        self.hits = 0
        self.misses = 0

    def get(self, key):
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        m = self.builder(key)
        self._cache[key] = m
        return m

    @property
    def size(self):
        return len(self._cache)


class GuidedDecoder:
    def __init__(self, tokenizer, validator, candidate_ids, eos_id=None,
                 vocab_size=None, device="cpu"):
        self.tok = tokenizer
        self.validator = validator
        self.candidates = list(candidate_ids)
        self.eos_id = eos_id
        self.vocab_size = vocab_size or (max(self.candidates) + 2)
        self.device = device
        self.cache = MaskCache(self._build)

    def _build(self, key):
        prefix, state = key
        ids = allowed_token_ids(self.tok, self.validator, prefix, self.candidates)
        if self.eos_id is not None and state == "valid":
            ids = ids + [self.eos_id]
        return build_mask(self.vocab_size, ids, self.device)

    def state(self, text):
        return self.validator(text)

    def mask_for(self, text):
        return self.cache.get((text, self.validator(text)))

    def is_complete(self, text):
        return self.validator(text) == "valid"
