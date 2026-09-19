"""Stage 04 (JAX) - static batching, right-padded.

`./vc lore 4 --jax` for the insight. `./vc test 4 --jax` to check yourself.

WHAT YOU ARE BUILDING

    static_batch_generate(model, prompts, max_tokens) -> list[list[int]]
    padding_waste(output_lens) -> float

One batch of N prompts. Prefill them together, then run a shared decode loop
until every row finishes. The output of each prompt must equal the output that
the prompt gives alone.

WHY THIS TRACK PADS ON THE RIGHT AND THE TORCH TRACK PADS ON THE LEFT

torch pads on the left, so that index -1 holds the last real token of every
row. The mask in HF hides the pad at the front.

You write into a cache, so the usual layout is the opposite. The prompt of
each row starts at slot 0 and goes to L_b. The pad is at the END, and the
generated tokens overwrite it.

So the last position is NOT the last token of every row. You must tell the
model where to read the logits of each row:

    logits, cache = model.forward(token_ids, positions, cache, cache_len,
                                  logits_index=jnp.array(prompt_lens) - 1)

    logits_index   (B,) the position of each row to read logits from -> (B, V)

If this is wrong, the short prompts of the batch continue the PAD. The text
looks fluent and has no meaning, so the error is not obvious. That is the
most frequent failure of this stage.

THE OUTLINE

    prompt_lens = [len(model.encode(prompt)) for prompt in prompts]
    longest     = max(prompt_lens)
    token_ids   = right-padded (B, longest)
    cache = model.init_cache(B, large enough)

    logits, cache = model.forward(token_ids, cache=cache, cache_len=zeros(B),
                                  logits_index=array(prompt_lens) - 1)
    cache_len = array(prompt_lens)        # for each row, NOT longest
    next_tokens = argmax(logits, -1)      # (B,)

    loop:
        record next_tokens for each row that still runs
        logits, cache = model.forward(next_tokens[:, None],
                                      positions=cache_len[:, None],
                                      cache=cache, cache_len=cache_len)
        cache_len = cache_len + 1
        next_tokens = argmax(logits, -1)

Note that `cache_len = prompt_lens`, not longest. For each short row, the
prefill wrote waste K and V into slots L_b to longest - 1. Set cache_len to
L_b. The first decoded token then overwrites that waste, and the mask never
reads it.

WHERE THE WASTE IS

There are two different wastes, and this stage shows both:

  - PREFILL waste: the batch pads every row to the longest prompt. So a
    batch of [12, 400] tokens does 800 tokens of prefill work for 412 tokens
    of prompt. That is the shape of the batch, and no layout corrects it.

  - DECODE waste: the batch runs until its SLOWEST row finishes. A row that
    got eos at step 3 keeps its slot for 60 more steps. Here it does not
    even get cheaper when it finishes, because the shape is fixed.
    `padding_waste` measures this waste.

Stage 05 fixes the second waste: it fills the slot again. Only one thing
fixes the first: do not batch prompts of very different lengths together.
You make that scheduling decision in stage 10.

    padding_waste(output_lens) -> 1 - sum(lens) / (num_rows * max(lens))
"""


def static_batch_generate(model, prompts: list[str],
                          max_tokens: int) -> list[list[int]]:
    """Prefill N prompts as one batch, then decode them together."""
    raise NotImplementedError("stage 04 (jax): implement static_batch_generate")


def padding_waste(output_lens: list[int]) -> float:
    """The fraction of decode slots spent on rows that already finished."""
    raise NotImplementedError("stage 04 (jax): implement padding_waste")
