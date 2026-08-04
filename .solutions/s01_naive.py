"""Reference solution, stage 01. Peek only after you've tried."""
import torch


@torch.inference_mode()
def naive_generate(model, tokenizer, prompt: str, max_tokens: int) -> list[int]:
    dev = next(model.parameters()).device
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(dev)
    eos = model.generation_config.eos_token_id
    eos = set(eos) if isinstance(eos, list) else {eos}
    out = []
    for _ in range(max_tokens):
        logits = model(ids, use_cache=False).logits
        nxt = int(logits[0, -1].argmax())
        if nxt in eos:
            break
        out.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=dev)], dim=1)
    return out
