"""Stage 04 (JAX) - static batching, right-padded.

`./vc lore 4 --jax` for the insight. `./vc test 4 --jax` to check yourself.

WHAT YOU'RE BUILDING

    static_batch_generate(model, prompts, max_tokens) -> list[list[int]]
    padding_waste(output_lens) -> float

One batch of N prompts, prefilled together, then a shared decode loop until
everyone is done. Each prompt's output must be what it would have been alone.

WHY THIS IS RIGHT-PADDED AND THE TORCH TRACK IS LEFT-PADDED

torch left-pads so that index -1 is everyone's last real token, because HF's
mask makes the leading pad invisible. You have a written-into cache instead, so
the natural layout is the other way round: every row's prompt starts at slot 0,
runs to L_b, and the pad sits at the END where the generated tokens are about
to overwrite it.

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

Note `cache_len = lens`, not T. The prefill wrote junk K/V into slots
L_b..T-1 for every short row, and setting cache_len to L_b is what makes the
first decoded token overwrite that junk and the mask never read it.

WHERE THE WASTE IS

Two different wastes, and this stage is about seeing both:

  - PREFILL waste: every row is padded out to the longest prompt, so a batch
    of [12, 400] tokens does 800 tokens of prefill work for 412 tokens of
    prompt. That is the batch's shape, and no layout fixes it.

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
