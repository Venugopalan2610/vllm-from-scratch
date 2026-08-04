"""Stage 01 - greedy decode, no cache.

`./vc lore` for the insight. `./vc test` to check yourself.
"""

import torch


@torch.inference_mode()
def naive_generate(model, tokenizer, prompt: str, max_tokens: int) -> list[int]:
    """Generate greedily, recomputing the entire prefix every single step.

    Returns the generated token ids only (not the prompt).

    Sketch:
        ids = tokenize(prompt)                           -> (1, L)
        for _ in range(max_tokens):
            logits = model(ids, use_cache=False).logits  -> (1, L, vocab)
            next_id = argmax(logits[0, -1])              # LAST position only
            if next_id is a stop token: break
            append next_id to ids and to the output list

    Two things that will bite you:
      - model.generation_config.eos_token_id may be a LIST of ids, not an int.
      - Do not use a cache here. Stage 02 measures its speedup against this
        number, and this is the one number you want to be embarrassing.
    """
    raise NotImplementedError("stage 01: implement naive_generate")
