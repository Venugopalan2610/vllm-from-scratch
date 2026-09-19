"""Stage 01 - greedy decode, no cache.

`./vc lore` for the insight. `./vc test` to check yourself.
"""

import torch


@torch.inference_mode()
def naive_generate(model, tokenizer, prompt: str, max_tokens: int) -> list[int]:
    """Generate greedily, and compute the whole prefix again at every step.

    Return only the generated token ids, not the prompt.

    Outline:
        token_ids = tokenize(prompt)                           -> (1, L)
        for _ in range(max_tokens):
            logits = model(token_ids, use_cache=False).logits  -> (1, L, vocab)
            next_token = argmax(logits[0, -1])                 # the LAST position
            if next_token is a stop token: break
            append next_token to token_ids and to the output list

    Two traps:
      - model.generation_config.eos_token_id can be a LIST of ids, not an int.
      - Do not use a cache here. Stage 02 measures its speedup against this
        number, and this number must be slow.
    """
    raise NotImplementedError("stage 01: implement naive_generate")
