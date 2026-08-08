"""Reference solution, stage 04 (jax)."""

import jax.numpy as jnp
import numpy as np

BUCKET = 256


def static_batch_generate(model, prompts: list[str],
                          max_tokens: int) -> list[list[int]]:
    enc = [[int(t) for t in model.encode(p)] for p in prompts]
    lens = np.array([len(e) for e in enc], dtype=np.int32)
    B, T = len(enc), int(lens.max())

    need = T + max_tokens + 1
    max_len = -(-need // BUCKET) * BUCKET

    # Right-padded. The pad sits where the generated tokens will land, and
    # cache_len=lens is what stops anything ever reading it.
    ids = np.zeros((B, T), dtype=np.int32)
    for i, e in enumerate(enc):
        ids[i, : len(e)] = e

    cache = model.init_cache(B, max_len)
    logits, cache = model.forward(
        jnp.asarray(ids), cache=cache, cache_len=jnp.zeros(B, jnp.int32),
        logits_index=jnp.asarray(lens - 1),
    )
    cache_len = jnp.asarray(lens)
    nxt = np.asarray(jnp.argmax(logits, axis=-1))

    outs = [[] for _ in range(B)]
    done = [False] * B

    for _ in range(max_tokens):
        for i, t in enumerate(nxt.tolist()):
            if done[i]:
                continue
            if t in model.eos_ids or len(outs[i]) >= max_tokens:
                done[i] = True
            else:
                outs[i].append(t)
                if len(outs[i]) >= max_tokens:
                    done[i] = True
        if all(done):
            break

        logits, cache = model.forward(
            jnp.asarray(nxt, jnp.int32)[:, None],
            positions=cache_len[:, None],
            cache=cache, cache_len=cache_len,
        )
        cache_len = cache_len + 1
        nxt = np.asarray(jnp.argmax(logits, axis=-1))

    return outs


def padding_waste(output_lens: list[int]) -> float:
    if not output_lens or max(output_lens) == 0:
        return 0.0
    useful = sum(output_lens)
    total = len(output_lens) * max(output_lens)
    return 1.0 - useful / total
