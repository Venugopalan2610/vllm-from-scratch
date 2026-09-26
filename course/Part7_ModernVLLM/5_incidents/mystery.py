"""Three speculative decoders. Each one has one fault.

DO NOT READ THIS FILE until you have written your diagnosis in
part7_inc_2_CCbreakItOnPurpose_helper.ipynb, Exercise 11.

Each takes (model, ids, max_tokens) like lab.speculative, and returns the new
tokens and a dict of statistics.
"""

import torch

from lab import ngram_propose, stop_ids


@torch.inference_mode()
def _spec(model, ids, max_tokens, k=4, keep_mismatch=False, crop=True, sequential=False):
    device = ids.device
    stops = stop_ids(model)
    context = ids[0].tolist()
    out = model(ids, use_cache=True)
    cache = out.past_key_values
    token = int(out.logits[0, -1].argmax())
    generated, steps, proposed, accepted = [], 0, 0, 0
    while len(generated) < max_tokens and token not in stops:
        generated.append(token)
        draft = ngram_propose(context + generated, k)
        if sequential:
            predicted = []
            for t in [token] + draft:
                logits = model(torch.tensor([[t]], device=device), past_key_values=cache, use_cache=True).logits
                predicted.append(int(logits[0, -1].argmax()))
        else:
            block = torch.tensor([[token] + draft], device=device)
            predicted = model(block, past_key_values=cache, use_cache=True).logits[0].argmax(-1).tolist()
        steps += 1
        proposed += len(draft)
        keep = []
        for d, p in zip(draft, predicted):
            if d != p:
                if keep_mismatch:
                    keep.append(d)
                break
            keep.append(d)
        accepted += len(keep)
        if crop:
            cache.crop(len(context) + len(generated) + len(keep))
        generated += keep
        token = predicted[min(len(keep), len(predicted) - 1)]
        for t in keep:
            if t in stops:
                generated = generated[:generated.index(t)]
                token = t
                break
    stats = dict(steps=steps, proposed=proposed, accepted=accepted,
                 tokens_per_step=round(len(generated) / max(steps, 1), 2))
    return generated[:max_tokens], stats


def spec_a(model, ids, max_tokens):
    # accept the draft up to and including the first token that differs
    return _spec(model, ids, max_tokens, keep_mismatch=True)


def spec_b(model, ids, max_tokens):
    # keep the cache as it is after the verify
    return _spec(model, ids, max_tokens, crop=False)


def spec_c(model, ids, max_tokens):
    # verify the draft tokens one by one, to keep the kernels simple
    return _spec(model, ids, max_tokens, sequential=True)
