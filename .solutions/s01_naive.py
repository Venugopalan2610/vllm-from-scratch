"""Reference solution, stage 01. Look only after you try it yourself."""
import torch


def stop_token_ids(model):
    stop = model.generation_config.eos_token_id
    return set(stop) if isinstance(stop, list) else {stop}


@torch.inference_mode()
def naive_generate(model, tokenizer, prompt: str, max_tokens: int) -> list[int]:
    device = next(model.parameters()).device
    token_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    stop_ids = stop_token_ids(model)
    generated = []
    for _ in range(max_tokens):
        # No cache: the whole prefix goes through the model again every step.
        logits = model(token_ids, use_cache=False).logits
        next_token = int(logits[0, -1].argmax())
        if next_token in stop_ids:
            break
        generated.append(next_token)
        next_column = torch.tensor([[next_token]], device=device)
        token_ids = torch.cat([token_ids, next_column], dim=1)
    return generated
