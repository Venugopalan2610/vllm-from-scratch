"""Reference solution, stage 04."""

import torch


def stop_token_ids(model):
    stop = model.generation_config.eos_token_id
    return set(stop) if isinstance(stop, list) else {stop}


def encode_left_padded(tokenizer, prompts, device):
    """Pad on the left, so that the last column is the last real token of
    every prompt."""
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    tokenizer.padding_side = old_side
    return encoded.input_ids, encoded.attention_mask


def collect_tokens(next_tokens, outputs, finished, stop_ids, max_tokens):
    """Append each new token to its row. Mark a row finished at a stop token
    or at max_tokens."""
    for row, token in enumerate(next_tokens.tolist()):
        if finished[row]:
            continue
        if token in stop_ids or len(outputs[row]) >= max_tokens:
            finished[row] = True
        else:
            outputs[row].append(token)


@torch.inference_mode()
def static_batch_generate(model, tokenizer, prompts: list[str],
                          max_tokens: int) -> list[list[int]]:
    device = next(model.parameters()).device
    stop_ids = stop_token_ids(model)
    token_ids, attention_mask = encode_left_padded(tokenizer, prompts, device)
    positions = (attention_mask.cumsum(-1) - 1).clamp(min=0)

    result = model(token_ids, attention_mask=attention_mask,
                   position_ids=positions, use_cache=True)
    next_tokens = result.logits[:, -1].argmax(-1)
    last_positions = positions[:, -1]

    batch_size = len(prompts)
    outputs = [[] for _ in range(batch_size)]
    finished = [False] * batch_size
    one_column = torch.ones(batch_size, 1, dtype=attention_mask.dtype,
                            device=device)

    for _ in range(max_tokens):
        collect_tokens(next_tokens, outputs, finished, stop_ids, max_tokens)
        if all(finished):
            break
        # A finished row still runs. That is the waste of a static batch.
        attention_mask = torch.cat([attention_mask, one_column], dim=1)
        last_positions = last_positions + 1
        result = model(next_tokens.unsqueeze(1), attention_mask=attention_mask,
                       position_ids=last_positions.unsqueeze(1),
                       past_key_values=result.past_key_values, use_cache=True)
        next_tokens = result.logits[:, -1].argmax(-1)
    return outputs


def padding_waste(output_lens: list[int]) -> float:
    if not output_lens or max(output_lens) == 0:
        return 0.0
    useful_tokens = sum(output_lens)
    computed_tokens = len(output_lens) * max(output_lens)
    return 1.0 - useful_tokens / computed_tokens
