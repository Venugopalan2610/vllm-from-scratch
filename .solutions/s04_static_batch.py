"""Reference solution, stage 04."""

import torch


def _eos_set(model):
    e = model.generation_config.eos_token_id
    return set(e) if isinstance(e, list) else {e}


@torch.inference_mode()
def static_batch_generate(model, tokenizer, prompts: list[str],
                          max_tokens: int) -> list[list[int]]:
    dev = next(model.parameters()).device
    eos = _eos_set(model)

    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(dev)
    tokenizer.padding_side = old_side

    ids, mask = enc.input_ids, enc.attention_mask
    pos = (mask.cumsum(-1) - 1).clamp(min=0)

    out = model(ids, attention_mask=mask, position_ids=pos, use_cache=True)
    past = out.past_key_values
    nxt = out.logits[:, -1].argmax(-1)
    cur = pos[:, -1]

    B = len(prompts)
    outs = [[] for _ in range(B)]
    done = [False] * B

    for _ in range(max_tokens):
        for i, t in enumerate(nxt.tolist()):
            if not done[i]:
                if t in eos or len(outs[i]) >= max_tokens:
                    done[i] = True
                else:
                    outs[i].append(t)
        if all(done):
            break
        mask = torch.cat(
            [mask, torch.ones(B, 1, dtype=mask.dtype, device=dev)], dim=1
        )
        cur = cur + 1
        out = model(nxt.unsqueeze(1), attention_mask=mask,
                    position_ids=cur.unsqueeze(1),
                    past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1)

    return outs


def padding_waste(output_lens: list[int]) -> float:
    if not output_lens or max(output_lens) == 0:
        return 0.0
    useful = sum(output_lens)
    total = len(output_lens) * max(output_lens)
    return 1.0 - useful / total
