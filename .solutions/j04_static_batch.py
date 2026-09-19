"""Reference solution, stage 04 (jax)."""

import jax.numpy as jnp
import numpy as np

CACHE_BUCKET = 256


def cache_length_for(longest_prompt, max_tokens):
    needed = longest_prompt + max_tokens + 1
    return -(-needed // CACHE_BUCKET) * CACHE_BUCKET


def right_padded(prompt_ids):
    """The pad sits where the generated tokens go. cache_len stops every read
    of it."""
    prompt_lens = np.array([len(ids) for ids in prompt_ids], dtype=np.int32)
    token_ids = np.zeros((len(prompt_ids), int(prompt_lens.max())),
                         dtype=np.int32)
    for row, ids in enumerate(prompt_ids):
        token_ids[row, :len(ids)] = ids
    return token_ids, prompt_lens


def collect_tokens(next_tokens, outputs, finished, stop_ids, max_tokens):
    for row, token in enumerate(next_tokens.tolist()):
        if finished[row]:
            continue
        if token in stop_ids or len(outputs[row]) >= max_tokens:
            finished[row] = True
            continue
        outputs[row].append(token)
        if len(outputs[row]) >= max_tokens:
            finished[row] = True


def static_batch_generate(model, prompts: list[str],
                          max_tokens: int) -> list[list[int]]:
    prompt_ids = [[int(token) for token in model.encode(prompt)]
                  for prompt in prompts]
    token_ids, prompt_lens = right_padded(prompt_ids)
    batch_size, longest_prompt = token_ids.shape

    cache = model.init_cache(batch_size,
                             cache_length_for(longest_prompt, max_tokens))
    logits, cache = model.forward(
        jnp.asarray(token_ids), cache=cache,
        cache_len=jnp.zeros(batch_size, jnp.int32),
        logits_index=jnp.asarray(prompt_lens - 1))
    cache_len = jnp.asarray(prompt_lens)
    next_tokens = np.asarray(jnp.argmax(logits, axis=-1))

    outputs = [[] for _ in range(batch_size)]
    finished = [False] * batch_size
    for _ in range(max_tokens):
        collect_tokens(next_tokens, outputs, finished, model.eos_ids,
                       max_tokens)
        if all(finished):
            break
        logits, cache = model.forward(
            jnp.asarray(next_tokens, jnp.int32)[:, None],
            positions=cache_len[:, None], cache=cache, cache_len=cache_len)
        cache_len = cache_len + 1
        next_tokens = np.asarray(jnp.argmax(logits, axis=-1))
    return outputs


def padding_waste(output_lens: list[int]) -> float:
    if not output_lens or max(output_lens) == 0:
        return 0.0
    useful_tokens = sum(output_lens)
    computed_tokens = len(output_lens) * max(output_lens)
    return 1.0 - useful_tokens / computed_tokens
