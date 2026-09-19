"""Stage 04 - static batching with padding.

`./vc lore 4` for the insight. `./vc test 4` to check yourself.
"""

import torch


@torch.inference_mode()
def static_batch_generate(model, tokenizer, prompts: list[str],
                          max_tokens: int) -> list[list[int]]:
    """Greedy-decode all the prompts in ONE batch. -> the generated ids of
    each prompt.

    The output must be the same tokens as cached_generate() on each prompt
    alone. Batching is an optimization, not a change of behavior.

    Outline:
        tokenizer.padding_side = "left"          # CRITICAL, see below
        encoded = tokenizer(prompts, return_tensors="pt", padding=True)
        positions = (encoded.attention_mask.cumsum(-1) - 1).clamp(min=0)
        result = model(encoded.input_ids, attention_mask=encoded.attention_mask,
                       position_ids=positions, use_cache=True)
        loop: give the argmax back as a (B, 1) tensor. At each step, add one
              column to the mask and add one to the position of each row.

    Three traps. Each one changes your output:

      - LEFT padding, not right. The last position of each row must be a
        real token, because you read the next-token logits there.
      - Give position_ids yourself. With left padding, the first real token
        of row i is not at index 0, and the model cannot find that from the
        ids. `(mask.cumsum(-1) - 1).clamp(min=0)` gives the correct answer.
      - Add one column to the attention_mask BEFORE each decode pass. If not,
        the rows attend to their own padding.

    A sequence that gets EOS must stop adding tokens. But its row stays in
    the batch until the whole batch finishes. That waste is the point of this
    stage. Measure it with padding_waste(), then remove it in stage 05.
    """
    raise NotImplementedError("stage 04: implement static_batch_generate")


def padding_waste(output_lens: list[int]) -> float:
    """The fraction of decode rows spent on sequences that already finished.

    A static batch runs until its LONGEST sequence finishes. Every sequence
    keeps its row for all max(output_lens) steps.

        useful = sum(output_lens)
        total  = len(output_lens) * max(output_lens)
        waste  = 1 - useful / total

    Return 0.0 for an empty list or for lengths that are all zero.
    """
    raise NotImplementedError("stage 04: implement padding_waste")
