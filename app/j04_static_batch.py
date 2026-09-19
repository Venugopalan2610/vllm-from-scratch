"""Stage 04 (JAX) - static batching, right-padded.

`./vc lore 4 --jax` for the insight. `./vc test 4 --jax` to check yourself.

WHAT YOU'RE BUILDING

    static_batch_generate(model, prompts, max_tokens) -> list[list[int]]
    padding_waste(output_lens) -> float

One batch of N prompts. Prefill them together, then run a shared decode loop
until every row finishes. The output of each prompt must equal the output that
the prompt gives alone.

WHY THIS TRACK PADS ON THE RIGHT AND THE TORCH TRACK PADS ON THE LEFT

torch pads on the left, so that index -1 holds the last real token of every
row. The mask in HF hides the pad at the front.

You write into a cache instead, so the natural layout is the opposite. The
prompt of each row starts at slot 0 and runs to L_b. The pad sits at the END,
and the generated tokens overwrite it.

Which means the last position is NOT everyone's last token, and you have to say
where each row's logits live:

    logits, cache = model.forward(ids, positions, cache, cache_len,
                                  logits_index=jnp.array([L - 1 for L in lens]))

    logits_index   (B,) per-row position to read logits from -> (B, V)

Get this wrong and short prompts in the batch generate a continuation of the
PAD, which reads as fluent nonsense rather than an obvious error. It is the
single most common way this stage fails.

THE SKETCH

    lens = [len(model.encode(p)) for p in prompts]
    T    = max(lens)
    ids  = right-padded (B, T)
    cache = model.init_cache(B, big enough)

    logits, cache = model.forward(ids, cache=cache, cache_len=zeros(B),
                                  logits_index=array(lens) - 1)
    cache_len = array(lens)             # per row, NOT T
    next = argmax(logits, -1)           # (B,)

    loop:
        record next for every row still running
        logits, cache = model.forward(next[:, None],
                                      positions=cache_len[:, None],
                                      cache=cache, cache_len=cache_len)
        cache_len = cache_len + 1
        next = argmax(logits, -1)

Note that `cache_len = lens`, and not T. For every short row, the prefill wrote
waste K and V into slots L_b to T-1. Set cache_len to L_b. The first decoded
token then overwrites that waste, and the mask never reads it.

WHERE THE WASTE IS

Two different wastes, and this stage is about seeing both:

  - PREFILL waste: the batch pads every row out to the longest prompt. So a
    batch of [12, 400] tokens does 800 tokens of prefill work for 412 tokens
    of prompt. That is the shape of the batch, and no layout corrects it.

  - DECODE waste: the batch runs until its SLOWEST member finishes. A row that
    hit eos on step 3 keeps occupying its slot for another 60 steps, and here
    it does not even get cheaper when it finishes, because the shape is fixed.
    `padding_waste` measures this one.

Stage 05 fixes the second by refilling the slot. Nothing fixes the first except
not batching wildly different prompt lengths together, which is a scheduling
decision you make in stage 10.

    padding_waste(output_lens) -> 1 - sum(lens) / (n_rows * max(lens))
"""


def static_batch_generate(model, prompts: list[str],
                          max_tokens: int) -> list[list[int]]:
    """Prefill N prompts as one batch, then decode them together."""
    raise NotImplementedError("stage 04 (jax): implement static_batch_generate")


def padding_waste(output_lens: list[int]) -> float:
    """Fraction of decoded token-slots wasted on rows that already finished."""
    raise NotImplementedError("stage 04 (jax): implement padding_waste")
