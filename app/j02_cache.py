"""Stage 02 (JAX) - the KV cache, preallocated.

`./vc lore 2 --jax` for the insight. `./vc test 2 --jax` to check yourself.

WHAT YOU ARE BUILDING

    cached_generate(model, prompt, max_tokens) -> list[int]
    kv_bytes_per_token(config) -> int

The rules are the same as on the torch track: one prefill over the whole
prompt, then ONE token for each forward pass. The output must be the same
tokens as stage 01.

THE PART THAT IS DIFFERENT

torch gives you a cache that is one token longer after each step. You cannot
do that here. XLA compiles for exact shapes, so a cache that changes shape
at every step costs a compile at every step. That made stage 01 so slow.

So the cache is a fixed buffer. You allocate it one time and write into it:

    cache = model.init_cache(batch=1, max_len=...)
      -> (keys, values), each (layers, batch, max_len, kv_heads, head_dim), zeros

    logits, cache = model.forward(token_ids, positions, cache, cache_len)

      cache_len   (B,) the number of slots that are ALREADY valid, before
                  this call. The model writes the K and V of this call at
                  cache_len .. cache_len + T - 1, and it masks attention to
                  slots <= cache_len + t.
      positions   the absolute RoPE positions. For the prefill that is
                  0..L-1. For a decode step at slot s it is [[s]]. If this
                  is wrong, the output changes after a few tokens. That is
                  the most frequent failure of this stage.

    You keep cache_len yourself. That is all the bookkeeping.

THE OUTLINE

    prompt_ids = model.encode(prompt)                     -> (L,)
    cache = model.init_cache(1, large enough)
    logits, cache = model.forward(prompt_ids[None], cache=cache, cache_len=[0])
    cache_len = L
    next_token = argmax(logits[0])
    loop:
        if next_token is eos: stop
        emit next_token
        logits, cache = model.forward([[next_token]], positions=[[cache_len]],
                                      cache=cache, cache_len=[cache_len])
        cache_len += 1
        next_token = argmax(logits[0])

TRAPS

  - The max_len of init_cache is a HARD limit, and a write past it does not
    raise an error. `dynamic_update_slice` clamps a start that is out of
    range. So a write past the end writes the last slot again and again,
    with no error, and the output becomes repetition. Make it large enough
    for the prompt + max_tokens.

  - If you make it EXACTLY prompt + max_tokens, each prompt length gives the
    decode step a different cache shape, so each new prompt costs one more
    compile. Round max_len up to a bucket (256, for example). Then prompts of
    200 and 250 tokens share one compiled decode step. This is your first
    look at stage 12, and here it costs almost nothing.

  - Make a new cache for each call. If you use one cache for two
    generate() calls and do not reset cache_len, the second request attends
    to the tokens of the first.

kv_bytes_per_token must compute, from a Qwen3Config:

    2 (K and V) * num_hidden_layers * num_key_value_heads * head_dim * dtype_bytes

The trap is num_key_value_heads. This model is GQA 16:8, so it has half as
many KV heads as attention heads. Other models have 8 times fewer. If you use
num_attention_heads by mistake, every memory budget from stage 06 on is wrong
by that factor.
"""


def cached_generate(model, prompt: str, max_tokens: int) -> list[int]:
    """Prefill one time, then one token for each forward pass. The output is
    the same as stage 01."""
    raise NotImplementedError("stage 02 (jax): implement cached_generate")


def kv_bytes_per_token(config) -> int:
    """The bytes of KV cache for one token, over all layers, in bf16."""
    raise NotImplementedError("stage 02 (jax): implement kv_bytes_per_token")
