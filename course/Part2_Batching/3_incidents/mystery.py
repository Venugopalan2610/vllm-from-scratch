"""Three batched loops. Each one has one fault.

DO NOT READ THIS FILE until you have written your diagnosis in
part2_inc_2_CCbreakItOnPurpose_helper.ipynb, Exercise 9.

Each loop takes a list of prompts and returns one list of token ids for each
prompt, in the order of the prompts.
"""

import torch
from transformers import DynamicCache

from lab import stop_ids


@torch.inference_mode()
def _batch(model, tokenizer, prompts, max_tokens, cache=None, stop_all_at_first=False):
    device = next(model.parameters()).device
    tokenizer.padding_side = 'left'
    batch = tokenizer(prompts, return_tensors='pt', padding=True).to(device)
    mask = batch.attention_mask
    stops = stop_ids(model)
    if cache is not None:
        mask = torch.cat([mask.new_ones(len(prompts), cache.get_seq_length()), mask], dim=1)
    out = model(batch.input_ids, attention_mask=mask, past_key_values=cache, use_cache=True)
    cache = out.past_key_values
    next_tokens = out.logits[:, -1].argmax(-1)
    generated = [[] for _ in prompts]
    done = [False] * len(prompts)
    for _ in range(max_tokens):
        for row, token in enumerate(next_tokens.tolist()):
            if not done[row]:
                if token in stops:
                    done[row] = True
                else:
                    generated[row].append(token)
        if all(done) or (stop_all_at_first and any(done)):
            break
        mask = torch.cat([mask, mask.new_ones(len(prompts), 1)], dim=1)
        out = model(next_tokens[:, None], attention_mask=mask, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        next_tokens = out.logits[:, -1].argmax(-1)
    return generated, cache


def mystery_a(model, tokenizer, prompts, max_tokens):
    # sort by length, so that the batch pads less
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    generated, _ = _batch(model, tokenizer, [prompts[i] for i in order], max_tokens)
    return generated                        # still in the sorted order


def mystery_b(model, tokenizer, prompts, max_tokens):
    # the loop ends when a row emits a stop token
    generated, _ = _batch(model, tokenizer, prompts, max_tokens, stop_all_at_first=True)
    return generated


def mystery_c(model, tokenizer, prompts, max_tokens):
    # at most 4 prompts in one forward pass, to bound the memory
    results, cache = [], DynamicCache()
    for start in range(0, len(prompts), 4):
        generated, cache = _batch(model, tokenizer, prompts[start:start + 4], max_tokens, cache=cache)
        results += generated
    return results
