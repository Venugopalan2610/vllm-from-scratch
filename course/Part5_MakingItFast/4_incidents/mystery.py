"""Three samplers. Each one has one fault.

DO NOT READ THIS FILE until you have written your diagnosis in
part5_inc_2_CCbreakItOnPurpose_helper.ipynb, Exercise 10.

Each sampler takes (logits, params, generator). logits: (batch, vocab).
params: one dict for each row, with temperature, top_k (0 is off), top_p,
penalty and seen. It returns one token id for each row.
"""

import torch


def _draw(probs, generator):
    noise = torch.empty_like(probs).exponential_(generator=generator)
    return int((probs / noise).argmax())


def _filter(probs, top_k, top_p):
    if top_k is not None:
        kth = probs.topk(top_k).values[-1] if top_k > 0 else float('inf')
        probs = probs.masked_fill(probs < kth, 0.0)
    sorted_probs, order = probs.sort(descending=True)
    before = sorted_probs.cumsum(-1) - sorted_probs
    sorted_probs = sorted_probs.masked_fill(before > top_p, 0.0)
    return torch.zeros_like(probs).scatter(-1, order, sorted_probs)


def _penalize(logits, penalty, seen):
    logits = logits.float().clone()
    for token in set(seen):
        logits[token] = logits[token] / penalty if logits[token] > 0 else logits[token] * penalty
    return logits


def sampler_a(logits, params, generator):
    # one temperature for the batch: the batch shares one softmax
    temperature = params[0]['temperature']
    out = []
    for row, p in enumerate(params):
        probs = torch.softmax(_penalize(logits[row], p['penalty'], p['seen']) / temperature, -1)
        out.append(_draw(_filter(probs, p['top_k'] or None, p['top_p']), generator))
    return torch.tensor(out, device=logits.device)


def sampler_b(logits, params, generator):
    # the repetition penalty divides the logit of a token that was seen
    out = []
    for row, p in enumerate(params):
        z = logits[row].float().clone()
        for token in set(p['seen']):
            z[token] = z[token] / p['penalty']
        probs = torch.softmax(z / p['temperature'], -1)
        out.append(_draw(_filter(probs, p['top_k'] or None, p['top_p']), generator))
    return torch.tensor(out, device=logits.device)


def sampler_c(logits, params, generator):
    # top_k is passed straight to the filter
    out = []
    for row, p in enumerate(params):
        probs = torch.softmax(_penalize(logits[row], p['penalty'], p['seen']) / p['temperature'], -1)
        out.append(_draw(_filter(probs, p['top_k'], p['top_p']), generator))
    return torch.tensor(out, device=logits.device)
