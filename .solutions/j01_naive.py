"""Reference solution, stage 01 (jax). Peek only after you've tried."""

import jax.numpy as jnp


def naive_generate(model, prompt: str, max_tokens: int) -> list[int]:
    ids = [int(t) for t in model.encode(prompt)]
    out = []
    for _ in range(max_tokens):
        # No cache argument at all: the model allocates a scratch cache exactly
        # as long as this prefix, fills it, and drops it. That is the quadratic
        # recompute, made visible.
        logits, _ = model.forward(jnp.asarray([ids], dtype=jnp.int32))
        nxt = int(logits[0].argmax())
        if nxt in model.eos_ids:
            break
        out.append(nxt)
        ids.append(nxt)
    return out
