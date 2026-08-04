"""Stage 04 - static batching with padding.

`./vc lore 4` for the insight. `./vc test 4` to check yourself.
"""

import torch


@torch.inference_mode()
def static_batch_generate(model, tokenizer, prompts: list[str],
                          max_tokens: int) -> list[list[int]]:
    """Greedy-decode all prompts in ONE batch. Returns generated ids per prompt.

    Output must be token-identical to calling cached_generate() on each prompt
    separately. Batching is an optimization, not a behavior change.

    Sketch:
        tokenizer.padding_side = "left"          # CRITICAL, see below
        enc  = tokenizer(prompts, return_tensors="pt", padding=True)
        pos  = (enc.attention_mask.cumsum(-1) - 1).clamp(min=0)
        out  = model(enc.input_ids, attention_mask=enc.attention_mask,
                     position_ids=pos, use_cache=True)
        loop: feed the argmax back as a (B, 1) tensor, growing the mask by one
              column each step and incrementing each row's position.

    Three traps, all of which change your output if you get them wrong:

      - LEFT padding, not right. Every row's last position must be a real
        token, because that is where you read the next-token logits from.
      - Pass position_ids explicitly. With left padding, row i's first real
        token is not at index 0, and the model cannot infer that from the ids.
        `(mask.cumsum(-1) - 1).clamp(min=0)` gives the right answer.
      - Extend the attention_mask by one column BEFORE each decode forward, or
        rows will attend to their own padding.

    A sequence that hits EOS must stop contributing tokens -- but its slot is
    still occupied until the whole batch finishes. That waste is the point of
    this stage; measure it with padding_waste(), then delete it in stage 05.
    """
    raise NotImplementedError("stage 04: implement static_batch_generate")


def padding_waste(output_lens: list[int]) -> float:
    """Fraction of decode slots burned on sequences that already finished.

    A static batch runs until its LONGEST member is done. Every sequence
    occupies its slot for all max(output_lens) steps regardless.

        useful = sum(output_lens)
        total  = len(output_lens) * max(output_lens)
        waste  = 1 - useful / total

    Return 0.0 for an empty list or all-zero lengths.
    """
    raise NotImplementedError("stage 04: implement padding_waste")
