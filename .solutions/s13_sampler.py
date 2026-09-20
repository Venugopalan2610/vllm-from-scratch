"""Reference solution, stage 13 - batched sampler."""

from dataclasses import dataclass

import torch

TINY = 1e-20


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: int = 0            # 0 = off
    top_p: float = 1.0        # 1.0 = off
    min_p: float = 0.0        # 0.0 = off
    repetition_penalty: float = 1.0
    seed: int | None = None
    logprobs: int | None = None  # None or number of top logprobs to return

    @property
    def greedy(self):
        return self.temperature == 0.0


def check_logits(logits):
    """Assert that no logit is NaN."""
    if torch.isnan(logits).any():
        raise ValueError(
            "NaN detected in logits. Common causes: bad quantization scale, "
            "FP8 calibration overflow, or attention overflow. Check your "
            "weights and KV cache.")


def apply_repetition_penalty(logits, prev_tokens, penalties):
    """logits (rows, vocab), prev_tokens list[list[int]], penalties (rows,)."""
    for row, tokens in enumerate(prev_tokens):
        if not tokens or penalties[row] == 1.0:
            continue
        seen = torch.tensor(sorted(set(tokens)), device=logits.device,
                            dtype=torch.long)
        seen_logits = logits[row, seen]
        logits[row, seen] = torch.where(seen_logits > 0,
                                        seen_logits / penalties[row],
                                        seen_logits * penalties[row])
    return logits


def apply_top_k(logits, k):
    """k (rows,). A 0 turns top-k off for that row."""
    if int(k.max()) <= 0:
        return logits
    sorted_logits, _ = torch.sort(logits, dim=-1, descending=True)
    kept_rank = k.clamp(min=1, max=logits.shape[1]) - 1
    threshold = sorted_logits.gather(1, kept_rank.unsqueeze(1))  # k-th largest
    keep = (k <= 0).unsqueeze(1) | (logits >= threshold)
    return torch.where(keep, logits, float("-inf"))


def apply_top_p(logits, p):
    """p (rows,). A 1.0 turns top-p off. Keep the smallest set of tokens whose
    cumulative probability reaches p. Always keep the top token."""
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_ids = torch.sort(probs, dim=-1, descending=True)
    mass_before = sorted_probs.cumsum(dim=-1) - sorted_probs
    # A token is not necessary if the mass before it already reaches p.
    remove_sorted = mass_before >= p.unsqueeze(1) - 1e-9
    remove_sorted[:, 0] = False
    remove = torch.zeros_like(remove_sorted).scatter(1, sorted_ids,
                                                     remove_sorted)
    return logits.masked_fill(remove, float("-inf"))


def apply_min_p(logits, min_p):
    """Filter out tokens whose probability is less than min_p * max_prob.
    min_p is (rows,). 0.0 turns it off."""
    if (min_p <= 0).all():
        return logits
    probs = torch.softmax(logits, dim=-1)
    max_prob = probs.max(dim=-1, keepdim=True).values
    threshold = max_prob * min_p.unsqueeze(1) - 1e-6
    mask = (probs < threshold) & (min_p.unsqueeze(1) > 0)
    # Always keep the top token
    mask.scatter_(1, probs.argmax(dim=-1, keepdim=True), False)
    return logits.masked_fill(mask, float("-inf"))


def row_values(params, field, device, dtype=torch.float32):
    return torch.tensor([getattr(p, field) for p in params], device=device,
                        dtype=dtype)


def uniform_noise(params, shape, device):
    """One draw for the batch. Only a row with a seed gets its own
    generator: a loop over every row costs one launch for each row."""
    noise = torch.rand(shape, device=device)
    for row, row_params in enumerate(params):
        if row_params.seed is not None:
            generator = torch.Generator(device=device)
            noise[row].uniform_(generator=generator.manual_seed(row_params.seed))
    return noise


def gumbel_noise(uniform):
    # Look at the parentheses. `-torch.log(x).clamp_min(e)` is
    # `-(torch.log(x).clamp_min(e))`. That clamps a NEGATIVE number to +e.
    # The outer log then gives NaN, and argmax returns 0 for ever.
    return -torch.log((-torch.log(uniform.clamp_min(TINY))).clamp_min(TINY))


def sample(logits, params, prev_tokens=None):
    """logits (rows, vocab) -> (rows,) token ids, in one vectorized pass."""
    check_logits(logits)
    device = logits.device
    scores = logits.float().clone()
    if prev_tokens is not None:
        scores = apply_repetition_penalty(
            scores, prev_tokens, row_values(params, "repetition_penalty", device))

    temperature = row_values(params, "temperature", device)
    greedy = temperature == 0
    scores = scores / torch.where(greedy, 1.0, temperature).unsqueeze(1)
    scores = apply_top_k(scores, row_values(params, "top_k", device, torch.long))
    scores = apply_top_p(scores, row_values(params, "top_p", device))
    scores = apply_min_p(scores, row_values(params, "min_p", device))

    # Gumbel-max: argmax(logits + Gumbel noise) is an exact categorical draw.
    # It vectorizes. torch.multinomial with a generator for each row does not.
    noise = gumbel_noise(uniform_noise(params, scores.shape, device))
    sampled = (scores + noise).argmax(dim=-1)
    return torch.where(greedy, scores.argmax(dim=-1), sampled)
