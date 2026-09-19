"""Reference solution, stage 02 (jax). Look only after you try it yourself."""

import jax.numpy as jnp

CACHE_BUCKET = 256


def cache_length_for(prompt_len, max_tokens):
    """Round the cache up to a bucket. With an exact size, each prompt length
    gives a new decode shape, and so a new XLA compilation."""
    needed = prompt_len + max_tokens + 1
    return -(-needed // CACHE_BUCKET) * CACHE_BUCKET


def cached_generate(model, prompt: str, max_tokens: int) -> list[int]:
    prompt_ids = model.encode(prompt)
    prompt_len = int(prompt_ids.shape[0])

    cache = model.init_cache(1, cache_length_for(prompt_len, max_tokens))
    logits, cache = model.forward(prompt_ids[None], cache=cache,
                                  cache_len=jnp.zeros(1, jnp.int32))
    cache_len = prompt_len
    next_token = int(logits[0].argmax())

    generated = []
    for _ in range(max_tokens):
        if next_token in model.eos_ids:
            break
        generated.append(next_token)
        logits, cache = model.forward(
            jnp.asarray([[next_token]], jnp.int32),
            positions=jnp.asarray([[cache_len]], jnp.int32), cache=cache,
            cache_len=jnp.asarray([cache_len], jnp.int32))
        cache_len += 1
        next_token = int(logits[0].argmax())
    return generated


def kv_bytes_per_token(config, dtype_bytes: int = 2) -> int:
    return (2 * config.num_hidden_layers * config.num_key_value_heads
            * config.head_dim * dtype_bytes)
