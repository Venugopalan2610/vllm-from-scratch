"""Stage 02 (JAX) - the KV cache, preallocated.

`./vc lore 2 --jax` for the insight. `./vc test 2 --jax` to check yourself.

WHAT YOU'RE BUILDING

    cached_generate(model, prompt, max_tokens) -> list[int]
    kv_bytes_per_token(config) -> int

Same rules as the torch track: one prefill over the whole prompt, then ONE
token per forward, output token-identical to stage 01.

THE PART THAT IS NOT THE SAME

torch hands you back a cache that has grown by one token. You cannot do that
here, because XLA compiles for exact shapes and a cache that changes shape
every step is a compile every step -- which is what made stage 01 so slow.

So the cache is a fixed buffer you allocate once and write into:

    cache = model.init_cache(batch=1, max_len=...)
      -> (k, v), each (layers, batch, max_len, kv_heads, head_dim), zeros

    logits, cache = model.forward(ids, positions, cache, cache_len)

      cache_len   (B,) how many slots are ALREADY valid, before this call.
                  The model writes this call's K/V at cache_len .. +T-1, and
                  masks attention to slots <= cache_len + t.
      positions   absolute RoPE positions. For the prefill that is 0..L-1; for
                  a decode step at slot s it is [[s]]. Get this wrong and the
                  output drifts a few tokens in, which is the single most
                  common way this stage fails.

    You keep cache_len yourself. It is the whole bookkeeping job.

THE SKETCH

    ids = model.encode(prompt)                      -> (L,)
    cache = model.init_cache(1, big enough)
    logits, cache = model.forward(ids[None], cache=cache, cache_len=[0])
    cache_len = L
    next = argmax(logits[0])
    loop:
        if next is eos: stop
        emit next
        logits, cache = model.forward([[next]], positions=[[cache_len]],
                                      cache=cache, cache_len=[cache_len])
        cache_len += 1
        next = argmax(logits[0])

TRAPS

  - init_cache's max_len is a HARD ceiling, and overflowing it does not raise.
    `dynamic_update_slice` clamps an out-of-range start, so writing past the
    end silently rewrites the last slot forever and the output degrades into
    repetition. Size it for prompt + max_tokens.

  - Size it to EXACTLY prompt + max_tokens and every distinct prompt length
    gives the decode step a distinct cache shape, so every new prompt costs
    another compile. Round max_len up to a bucket (256, say) and prompts of
    200 and 250 tokens share one compiled decode step. This is your first
    taste of stage 12, and it is nearly free to do here.

  - A fresh cache per call. Reuse one across two generate() calls without
    resetting cache_len and the second request attends to the first's tokens.

kv_bytes_per_token must compute, from a Qwen3Config:

    2 (K and V) * num_hidden_layers * num_key_value_heads * head_dim * dtype_bytes

The trap is num_key_value_heads. This model is GQA 16:8, so KV heads are half
the attention heads; other models are 8x fewer. Use num_attention_heads by
mistake and every memory budget from stage 06 onward is wrong by that factor.
"""


def cached_generate(model, prompt: str, max_tokens: int) -> list[int]:
    """Prefill once, then one token per forward. Same output as stage 01."""
    raise NotImplementedError("stage 02 (jax): implement cached_generate")


def kv_bytes_per_token(config) -> int:
    """Bytes of KV cache one token costs, across all layers, in bf16."""
    raise NotImplementedError("stage 02 (jax): implement kv_bytes_per_token")
