"""Reference solution, stage 02. Peek only after you've tried."""
import torch


@torch.inference_mode()
def cached_generate(model, tokenizer, prompt: str, max_tokens: int) -> list[int]:
    dev = next(model.parameters()).device
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(dev)
    eos = model.generation_config.eos_token_id
    eos = set(eos) if isinstance(eos, list) else {eos}

    o = model(ids, use_cache=True)          # prefill
    past = o.past_key_values
    nxt = int(o.logits[0, -1].argmax())

    out = []
    for _ in range(max_tokens):
        if nxt in eos:
            break
        out.append(nxt)
        o = model(torch.tensor([[nxt]], device=dev), past_key_values=past, use_cache=True)
        past = o.past_key_values
        nxt = int(o.logits[0, -1].argmax())
    return out


def kv_bytes_per_token(config) -> int:
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    return 2 * config.num_hidden_layers * config.num_key_value_heads * head_dim * 2
