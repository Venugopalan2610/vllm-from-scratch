"""Reference solution, stage 02 (jax). Peek only after you've tried."""

import jax.numpy as jnp

BUCKET = 256


def cached_generate(model, prompt: str, max_tokens: int) -> list[int]:
    ids = model.encode(prompt)
    L = int(ids.shape[0])

    # Round the cache up to a bucket. Sized exactly, every distinct prompt
    # length would be a distinct decode-step shape, and therefore a distinct
    # XLA compilation.
    need = L + max_tokens + 1
    max_len = -(-need // BUCKET) * BUCKET

    cache = model.init_cache(1, max_len)
    logits, cache = model.forward(ids[None], cache=cache,
                                  cache_len=jnp.zeros(1, jnp.int32))
    cache_len = L
    nxt = int(logits[0].argmax())

    out = []
    for _ in range(max_tokens):
        if nxt in model.eos_ids:
            break
        out.append(nxt)
        pos = jnp.asarray([[cache_len]], jnp.int32)
        logits, cache = model.forward(
            jnp.asarray([[nxt]], jnp.int32), positions=pos, cache=cache,
            cache_len=jnp.asarray([cache_len], jnp.int32),
        )
        cache_len += 1
        nxt = int(logits[0].argmax())
    return out


def kv_bytes_per_token(config, dtype_bytes: int = 2) -> int:
    return (2 * config.num_hidden_layers * config.num_key_value_heads
            * config.head_dim * dtype_bytes)
