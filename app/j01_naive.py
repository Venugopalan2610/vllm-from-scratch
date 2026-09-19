"""Stage 01 (JAX) - greedy decode, no cache.

`./vc lore 1 --jax` for the insight. `./vc test 1 --jax` to check yourself.

Read `jvllm/model.py` first. It is the JAX Qwen3 that this track builds on,
and the repo gives it to you. You do not write a transformer here, the same
way that the torch track does not write one.

WHAT YOU ARE BUILDING

    naive_generate(model, prompt, max_tokens) -> list[int]

It returns the GENERATED token ids, and not the prompt.

The rules are the same as on the torch track:

  - Greedy. Always take the argmax of the logits at the final position.
  - NO cache. Every step runs the model again over the whole prefix.
  - Stop early on eos.

    model.encode(text) -> (T,) int32        model.decode(ids) -> str
    model.eos_ids      -> set[int]          NOTE: several ids, not one
    model.forward(ids) -> (logits, cache)   ids (B, T), logits (B, V) by default

TWO THINGS HURT HERE, AND ONLY ONE OF THEM IS THE ALGORITHM

The quadratic recompute is the point of the stage. The torch track pays the
same cost.

The other one is new. Each new prompt length is a new SHAPE, and each new
shape is an XLA compilation. That costs a second or more, on the critical
path, before any arithmetic happens.

A naive run of 64 tokens sees 64 lengths, so it does 64 compiles. The
compiler, and not the GPU, fills most of your first run. A utilisation graph
does not tell you that.

You do not fix it here. Stage 02 removes it almost by accident, because a
preallocated cache has ONE shape. Stage 12 is about it directly. But measure
it now, so that you know the shape of the problem later:

    from jvllm.model import _forward
    _forward._cache_size()      how many shapes it compiled

One more point. `int(logits[0].argmax())` forces a device-to-host sync at
every step. In this stage you cannot avoid it, because you need the token to
build the next input. But know that it is a sync. That is why the step loop in
stage 05 does exactly one sync for each iteration, and not one for each
sequence.
"""


def naive_generate(model, prompt: str, max_tokens: int) -> list[int]:
    """Greedy decode with no cache. -> the generated token ids.

    Outline:
        token_ids = [int(token) for token in model.encode(prompt)]
        for _ in range(max_tokens):
            logits, _ = model.forward(jnp.asarray([token_ids], jnp.int32))
            next_token = int(logits[0].argmax())     # the LAST position only
            if next_token in model.eos_ids:
                break
            generated.append(next_token)
            token_ids.append(next_token)
    """
    raise NotImplementedError("stage 01 (jax): implement naive_generate")
