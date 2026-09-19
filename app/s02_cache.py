"""Stage 02 - the KV cache.

`./vc lore` for the insight. `./vc test` to check yourself.
"""

import torch


@torch.inference_mode()
def cached_generate(model, tokenizer, prompt: str, max_tokens: int) -> list[int]:
    """The same tokens as stage 01, with O(N) work, not O(N^2).

    Outline:
        prefill = model(prompt_ids, use_cache=True)   # PREFILL: the prompt, one time
        cache = prefill.past_key_values
        next_token = argmax(prefill.logits[0, -1])
        loop:
            decode = model(next_token.view(1, 1), past_key_values=cache,
                           use_cache=True)           # DECODE: one token each pass
            cache = decode.past_key_values
            next_token = argmax(decode.logits[0, -1])

    Two traps:
      - Make a NEW cache for each call. A cache from an earlier call corrupts
        the next request, and a check tests for that.
      - The model must know the absolute position of the new token. HF gets
        it from the length of the cache. When you write your own attention in
        stage 07, you must give it yourself. Note that dependency now.
    """
    raise NotImplementedError("stage 02: implement cached_generate")


def kv_bytes_per_token(config) -> int:
    """The bytes of KV cache for one token, for a bf16 cache.

        2 (K and V) * layers * num_key_value_heads * head_dim * 2 bytes

    Read `num_key_value_heads`, NOT `num_attention_heads`. Under GQA they are
    different by 4 to 8 times, and every memory budget from stage 06 on uses
    this number. head_dim can be a field of the config, or hidden_size //
    num_attention_heads.
    """
    raise NotImplementedError("stage 02: implement kv_bytes_per_token")
