"""Reference solution, stage 01 (jax). Look only after you try it yourself."""

import jax.numpy as jnp


def naive_generate(model, prompt: str, max_tokens: int) -> list[int]:
    token_ids = [int(token) for token in model.encode(prompt)]
    generated = []
    for _ in range(max_tokens):
        # No cache argument: the model makes a scratch cache as long as this
        # prefix, fills it, and drops it. That is the quadratic recompute.
        logits, _ = model.forward(jnp.asarray([token_ids], dtype=jnp.int32))
        next_token = int(logits[0].argmax())
        if next_token in model.eos_ids:
            break
        generated.append(next_token)
        token_ids.append(next_token)
    return generated
