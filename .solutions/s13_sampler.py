"""Reference solution, stage 13 - batched sampler."""

from dataclasses import dataclass

import torch


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: int = 0            # 0 = disabled
    top_p: float = 1.0        # 1.0 = disabled
    repetition_penalty: float = 1.0
    seed: int | None = None

    @property
    def greedy(self):
        return self.temperature == 0.0


def apply_repetition_penalty(logits, prev_tokens, penalties):
    """logits (B,V); prev_tokens list[list[int]]; penalties (B,)"""
    for i, toks in enumerate(prev_tokens):
        if not toks or penalties[i] == 1.0:
            continue
        idx = torch.tensor(sorted(set(toks)), device=logits.device, dtype=torch.long)
        vals = logits[i, idx]
        logits[i, idx] = torch.where(vals > 0, vals / penalties[i], vals * penalties[i])
    return logits


def apply_top_k(logits, k):
    """k is (B,) -- 0 means disabled for that row."""
    B, V = logits.shape
    out = logits
    kmax = int(k.max())
    if kmax <= 0:
        return out
    vals, _ = torch.sort(logits, dim=-1, descending=True)
    kk = k.clamp(min=1, max=V)
    # threshold = the k-th largest value in each row
    thresh = vals.gather(1, (kk - 1).unsqueeze(1))
    disabled = (k <= 0).unsqueeze(1)
    return torch.where(disabled | (logits >= thresh), out, float("-inf"))


def apply_top_p(logits, p):
    """p is (B,) -- 1.0 means disabled. Keeps the smallest set of tokens whose
    cumulative probability reaches p, always keeping at least one."""
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cum = sorted_probs.cumsum(dim=-1)
    # a token is unnecessary if the mass BEFORE it already reached p
    remove = (cum - sorted_probs) >= p.unsqueeze(1) - 1e-9
    remove[:, 0] = False                       # always keep the top token
    mask = torch.zeros_like(remove).scatter(1, sorted_idx, remove)
    return logits.masked_fill(mask, float("-inf"))


def sample(logits, params, prev_tokens=None):
    """logits (B, V) -> (B,) token ids. One vectorized pass over the batch."""
    B, V = logits.shape
    dev = logits.device
    out = logits.float().clone()

    if prev_tokens is not None:
        pen = torch.tensor([p.repetition_penalty for p in params], device=dev)
        out = apply_repetition_penalty(out, prev_tokens, pen)

    temp = torch.tensor([p.temperature for p in params], device=dev)
    greedy = temp == 0
    safe_temp = torch.where(greedy, torch.ones_like(temp), temp)
    out = out / safe_temp.unsqueeze(1)

    k = torch.tensor([p.top_k for p in params], device=dev, dtype=torch.long)
    out = apply_top_k(out, k)
    pp = torch.tensor([p.top_p for p in params], device=dev)
    out = apply_top_p(out, pp)

    # Gumbel-max: argmax(logits + Gumbel noise) is an exact categorical draw,
    # and it vectorises where torch.multinomial with per-row generators does not.
    u = torch.empty(B, V, device=dev)
    for i, p in enumerate(params):
        if p.seed is not None:
            g = torch.Generator(device=dev).manual_seed(p.seed)
            u[i].uniform_(generator=g)
        else:
            u[i].uniform_()
    # NOTE the parentheses: `-torch.log(x).clamp_min(e)` parses as
    # `-(torch.log(x).clamp_min(e))`, which clamps a NEGATIVE number to +e,
    # then takes log of a negative -> NaN -> argmax returns 0 forever.
    gumbel = -torch.log((-torch.log(u.clamp_min(1e-20))).clamp_min(1e-20))
    sampled = (out + gumbel).argmax(dim=-1)

    return torch.where(greedy, out.argmax(dim=-1), sampled)
