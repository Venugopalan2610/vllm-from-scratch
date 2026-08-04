"""Stage 02 - the KV cache.

`./vc lore` for the insight. `./vc test` to check yourself.
"""

import torch


@torch.inference_mode()
def cached_generate(model, tokenizer, prompt: str, max_tokens: int) -> list[int]:
    """Same tokens as stage 01, but O(N) work instead of O(N^2).

    Sketch:
        out  = model(prompt_ids, use_cache=True)   # PREFILL: whole prompt, once
        past = out.past_key_values
        nxt  = argmax(out.logits[0, -1])
        loop:
            out  = model(nxt.view(1,1), past_key_values=past, use_cache=True)
            past = out.past_key_values             # DECODE: one token per forward
            nxt  = argmax(out.logits[0, -1])

    Two traps:
      - Build a FRESH cache per call. A leaked cache corrupts the next request,
        and there is a test for exactly that.
      - The model must know the absolute position of the new token. HF infers it
        from the cache length. When you write your own attention in stage 07 you
        will have to pass it explicitly -- notice the dependency now.
    """
    raise NotImplementedError("stage 02: implement cached_generate")


def kv_bytes_per_token(config) -> int:
    """Bytes of KV cache that one token costs, for a bf16 cache.

        2 (K and V) * layers * num_key_value_heads * head_dim * 2 bytes

    Read `num_key_value_heads`, NOT `num_attention_heads`. Under GQA they differ
    by 4-8x, and this number is the basis of every memory budget from stage 06
    on. head_dim may be an explicit config field, or hidden_size //
    num_attention_heads.
    """
    raise NotImplementedError("stage 02: implement kv_bytes_per_token")
