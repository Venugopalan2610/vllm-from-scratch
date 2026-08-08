"""Stage 01 (JAX) - greedy decode, no cache.

`./vc lore 1 --jax` for the insight. `./vc test 1 --jax` to check yourself.

Read `jvllm/model.py` first. It is the JAX Qwen3 this track is built on, and it
is provided -- you do not write a transformer here, the same way the torch
track does not write one.

WHAT YOU'RE BUILDING

    naive_generate(model, prompt, max_tokens) -> list[int]

returning the GENERATED token ids, not the prompt.

Rules, identical to the torch track:

  - Greedy. Always argmax of the final position's logits.
  - NO cache. Every step re-runs the model over the whole prefix.
  - Stop early on eos.

    model.encode(text) -> (T,) int32        model.decode(ids) -> str
    model.eos_ids      -> set[int]          NOTE: several ids, not one
    model.forward(ids) -> (logits, cache)   ids (B, T); logits (B, V) by default

TWO THINGS BITE HERE, AND ONLY ONE OF THEM IS THE ALGORITHM

The quadratic recompute is the point of the stage, and it is the same cost the
torch track pays.

The other one is new. Every distinct prompt length is a distinct SHAPE, and
every distinct shape is an XLA compilation -- a second or more, on the critical
path, before any arithmetic happens. Generating 64 tokens naively means 64
lengths, so 64 compiles. Your first run of this will be dominated by the
compiler, not the GPU, and no amount of staring at a utilisation graph will
tell you that.

You are not asked to fix it here. Stage 02 kills it almost by accident (a
preallocated cache has ONE shape), and stage 12 is about it directly. But
measure it now, so you recognise the shape of the problem later:

    from jvllm.model import _forward
    _forward._cache_size()      how many shapes it has compiled for

One more: `int(logits[0].argmax())` forces a device-to-host sync every step. In
this stage that is unavoidable -- you need the token to build the next input.
Notice that it is a sync, though. It is why stage 05's step loop is written to
do exactly one of them per iteration and not one per sequence.
"""


def naive_generate(model, prompt: str, max_tokens: int) -> list[int]:
    """Greedy decode with no cache. Returns generated token ids.

    Sketch:
        ids = [int(t) for t in model.encode(prompt)]
        for _ in range(max_tokens):
            logits, _ = model.forward(jnp.asarray([ids], jnp.int32))
            nxt = int(logits[0].argmax())        # LAST position only
            if nxt in model.eos_ids: break
            out.append(nxt); ids.append(nxt)
    """
    raise NotImplementedError("stage 01 (jax): implement naive_generate")
