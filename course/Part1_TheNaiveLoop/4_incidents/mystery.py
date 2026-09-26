"""Three generation loops. Each one has one fault.

DO NOT READ THIS FILE until you have written your diagnosis in
part1_inc_2_CCbreakItOnPurpose_helper.ipynb, Exercise 11.

In a real incident you do not get the source of the broken thing first. You
get its behaviour. Find the fault from the behaviour. Then open this file and
check your answer.
"""

import torch
from transformers import DynamicCache


def _stop_ids(model):
    stop = model.generation_config.eos_token_id
    return set(stop) if isinstance(stop, list) else {stop}


# a module-level cache. It survives from one request to the next.
_cache = None


@torch.inference_mode()
def mystery_a(model, tokenizer, prompt: str, max_tokens: int) -> list[int]:
    global _cache
    if _cache is None:
        _cache = DynamicCache()
    device = next(model.parameters()).device
    token_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    stop_ids = _stop_ids(model)
    output = model(token_ids, past_key_values=_cache, use_cache=True)
    next_token = int(output.logits[0, -1].argmax())
    generated = []
    for _ in range(max_tokens):
        if next_token in stop_ids:
            break
        generated.append(next_token)
        output = model(torch.tensor([[next_token]], device=device),
                       past_key_values=_cache, use_cache=True)
        next_token = int(output.logits[0, -1].argmax())
    return generated


@torch.inference_mode()
def mystery_b(model, tokenizer, prompt: str, max_tokens: int) -> list[int]:
    device = next(model.parameters()).device
    token_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    stop_ids = _stop_ids(model)
    output = model(token_ids, use_cache=True)
    cache = output.past_key_values
    next_token = int(output.logits[0, -1].argmax())
    # the position of the next token. It starts one too far.
    position = token_ids.shape[1] + 1
    generated = []
    for _ in range(max_tokens):
        if next_token in stop_ids:
            break
        generated.append(next_token)
        output = model(torch.tensor([[next_token]], device=device),
                       past_key_values=cache, use_cache=True,
                       position_ids=torch.tensor([[position]], device=device))
        cache = output.past_key_values
        next_token = int(output.logits[0, -1].argmax())
        position += 1
    return generated


@torch.inference_mode()
def mystery_c(model, tokenizer, prompt: str, max_tokens: int) -> list[int]:
    device = next(model.parameters()).device
    # a limit copied from a classifier project. It cuts the END of the prompt.
    token_ids = tokenizer(prompt, return_tensors='pt', truncation=True,
                          max_length=24).input_ids.to(device)
    stop_ids = _stop_ids(model)
    output = model(token_ids, use_cache=True)
    cache = output.past_key_values
    next_token = int(output.logits[0, -1].argmax())
    generated = []
    for _ in range(max_tokens):
        if next_token in stop_ids:
            break
        generated.append(next_token)
        output = model(torch.tensor([[next_token]], device=device),
                       past_key_values=cache, use_cache=True)
        cache = output.past_key_values
        next_token = int(output.logits[0, -1].argmax())
    return generated
