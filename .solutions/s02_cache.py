"""Reference solution, stage 02. Look only after you try it yourself."""
import torch


def stop_token_ids(model):
    stop = model.generation_config.eos_token_id
    return set(stop) if isinstance(stop, list) else {stop}


@torch.inference_mode()
def cached_generate(model, tokenizer, prompt: str, max_tokens: int) -> list[int]:
    device = next(model.parameters()).device
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    stop_ids = stop_token_ids(model)

    prefill = model(prompt_ids, use_cache=True)
    cache = prefill.past_key_values
    next_token = int(prefill.logits[0, -1].argmax())

    generated = []
    for _ in range(max_tokens):
        if next_token in stop_ids:
            break
        generated.append(next_token)
        decode = model(torch.tensor([[next_token]], device=device),
                       past_key_values=cache, use_cache=True)
        cache = decode.past_key_values
        next_token = int(decode.logits[0, -1].argmax())
    return generated


def kv_bytes_per_token(config) -> int:
    """K and V, for every layer and every KV head, in bf16 (2 bytes)."""
    head_dim = (getattr(config, "head_dim", None)
                or config.hidden_size // config.num_attention_heads)
    bytes_per_value = 2
    return (2 * config.num_hidden_layers * config.num_key_value_heads
            * head_dim * bytes_per_value)
