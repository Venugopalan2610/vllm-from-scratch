"""Stage 13 (JAX) - a real batched sampler.

`./vc lore 13 --jax` for the insight. `./vc test 13 --jax` to check yourself.

Every request has its own temperature, top-k, top-p, penalties and seed, and
they all have to be applied in ONE vectorized pass over the batch. The naive
per-request Python loop quietly becomes your bottleneck the moment the kernels
are fast.

WHAT YOU'RE BUILDING

    @dataclass SamplingParams(temperature=1.0, top_k=0, top_p=1.0,
                              repetition_penalty=1.0, seed=None)
        .greedy -> temperature == 0

    apply_repetition_penalty(logits, prev_tokens, penalties) -> logits
    apply_top_k(logits, k) -> logits          k (B,) ints, 0 = disabled
    apply_top_p(logits, p) -> logits          p (B,) floats, 1.0 = disabled
    row_keys(params, key=None) -> (B,) PRNG keys
    sample(logits, params, prev_tokens=None, key=None) -> (B,) int32

Order matters, and it is the order every real server uses:

    repetition penalty -> temperature -> top-k -> top-p -> draw

RANDOMNESS IS THE INTERESTING PART HERE

JAX has no global RNG. There is no `manual_seed`, no hidden state, nothing a
concurrent request can perturb. A key is a value you carry.

That is a nuisance in a notebook and exactly right in a server. A request's
seed IS its key. Two requests in the same batch cannot affect each other's
stream, and a seeded request produces the same tokens whether it was batched
with one neighbour or sixty-three. Reproducibility stops being a promise you
make and becomes a property of the data structure.

    jax.random.key(seed)            a request that brought its own seed
    jax.random.fold_in(base, i)     one that did not

Then draw the whole batch at once with GUMBEL-MAX:

    argmax(logits + Gumbel noise)

is an exact categorical sample, and it vmaps cleanly over per-row keys, which
a per-row categorical does not.

    jax.vmap(lambda k: jax.random.gumbel(k, (V,)))(keys)

TRAPS

  - Repetition penalty: DIVIDE positive logits, MULTIPLY negative ones.
    Dividing a negative logit makes it bigger, which rewards exactly the token
    you meant to punish. This is a real bug that shipped in real servers.

  - Temperature 0 means greedy, and dividing by it gives inf/nan. Substitute a
    safe 1.0 for those rows and select argmax for them at the end -- do not
    branch, the whole point is one pass.

  - top-p keeps the smallest prefix whose cumulative mass REACHES p. A token is
    unnecessary when the mass BEFORE it already reached p. Always keep the top
    token, or a peaked distribution with a small p returns nothing.

  - jnp.sort is ascending and there is no `descending=`. Reverse it.

  - JIT THE CORE. Run eagerly, this sampler costs ~40 ms at batch 64 -- more
    than a decode step -- and essentially none of that is arithmetic. It is one
    device dispatch per jnp op, sixty-four separate PRNG key constructions, and
    a host round trip for anything you called int() on. Assemble the per-row
    parameters into arrays, then hand them to one jitted function. Which also
    means no `if int(k.max()) <= 0: return logits` shortcut: that is a sync,
    and it makes the function untraceable.

  - Build the keys as a batch too. `[jax.random.key(p.seed) for p in params]`
    is 64 dispatches. `jax.random.key_data` / `wrap_key_data` let you select
    between two vmapped fold_ins with an ordinary jnp.where.

  - `logits.at[i, idx].set(...)` returns a new array. Rebind it. A dropped
    `out =` here silently disables the penalty and nothing errors.
"""


class SamplingParams:
    """Per-request sampling configuration."""


def apply_repetition_penalty(logits, prev_tokens, penalties):
    """Penalise tokens each row has already emitted."""
    raise NotImplementedError("stage 13 (jax): implement apply_repetition_penalty")


def apply_top_k(logits, k):
    """Mask all but each row's top-k logits. k (B,), 0 disables that row."""
    raise NotImplementedError("stage 13 (jax): implement apply_top_k")


def apply_top_p(logits, p):
    """Nucleus mask. p (B,), 1.0 disables that row."""
    raise NotImplementedError("stage 13 (jax): implement apply_top_p")


def row_keys(params, key=None):
    """One PRNG key per request: its own seed, or folded off `key`."""
    raise NotImplementedError("stage 13 (jax): implement row_keys")


def sample(logits, params, prev_tokens=None, key=None):
    """One vectorized pass: penalty, temperature, top-k, top-p, draw."""
    raise NotImplementedError("stage 13 (jax): implement sample")
