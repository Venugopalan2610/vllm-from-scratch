"""Reference solution, stage 13 (jax) - batched sampler."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

# Seeded rows fold off a FIXED base, so a request's key is a function of its
# seed alone and nothing else in the batch can move it.
_SEED_BASE = jax.random.key(0)


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: int = 0            # 0 = disabled
    top_p: float = 1.0        # 1.0 = disabled
    repetition_penalty: float = 1.0
    seed: int | None = None

    @property
    def greedy(self):
        return self.temperature == 0.0


def apply_repetition_penalty(logits, prev_tokens, penalties):
    """logits (B,V); prev_tokens list[list[int]]; penalties (B,)

    Ragged per row, so this one stays outside jit -- which is fine, because it
    touches a handful of indices per row rather than the whole vocab.
    """
    out = logits
    for i, toks in enumerate(prev_tokens):
        pen = float(penalties[i])
        if not toks or pen == 1.0:
            continue
        idx = jnp.asarray(sorted(set(toks)), dtype=jnp.int32)
        vals = out[i, idx]
        # Divide positives, MULTIPLY negatives. Dividing a negative logit makes
        # it larger, i.e. rewards the token you meant to punish.
        out = out.at[i, idx].set(jnp.where(vals > 0, vals / pen, vals * pen))
    return out


def apply_top_k(logits, k):
    """k is (B,) -- 0 means disabled for that row.

    No `if int(k.max()) <= 0: return` shortcut. That reads one device value on
    the host, which is a sync, and it makes the whole function untraceable.
    The `disabled` term below does the same job for free.
    """
    B, V = logits.shape
    ordered = jnp.sort(logits, axis=-1)[:, ::-1]      # jnp.sort is ascending
    kk = jnp.clip(k, 1, V)
    thresh = jnp.take_along_axis(ordered, (kk - 1)[:, None], axis=1)
    disabled = (k <= 0)[:, None]
    return jnp.where(disabled | (logits >= thresh), logits, -jnp.inf)


def apply_top_p(logits, p):
    """Keep the smallest set of tokens whose cumulative probability reaches p,
    always keeping at least one."""
    probs = jax.nn.softmax(logits, axis=-1)
    order = jnp.argsort(probs, axis=-1)[:, ::-1]
    sorted_probs = jnp.take_along_axis(probs, order, axis=1)
    cum = jnp.cumsum(sorted_probs, axis=-1)
    # a token is unnecessary if the mass BEFORE it already reached p
    remove = (cum - sorted_probs) >= p[:, None] - 1e-9
    remove = remove.at[:, 0].set(False)               # always keep the top token
    mask = jnp.zeros_like(remove).at[
        jnp.arange(logits.shape[0])[:, None], order].set(remove)
    return jnp.where(mask, -jnp.inf, logits)


def row_keys(params, key=None):
    """One PRNG key per request, built with two vmapped fold_ins and a select.

    A Python loop over 64 requests would be 64 separate device dispatches
    before any sampling happens -- which, measured, is most of the sampler's
    wall clock. Keys are values; batch them like any other value.
    """
    base = _SEED_BASE if key is None else key
    seeds = jnp.asarray([-1 if p.seed is None else p.seed for p in params],
                        jnp.int32)
    idx = jnp.arange(len(params), dtype=jnp.int32)

    seeded = jax.vmap(lambda s: jax.random.fold_in(_SEED_BASE, s))(seeds)
    unseeded = jax.vmap(lambda i: jax.random.fold_in(base, i))(idx)
    data = jnp.where((seeds >= 0)[:, None],
                     jax.random.key_data(seeded),
                     jax.random.key_data(unseeded))
    return jax.random.wrap_key_data(data)


@jax.jit
def _sample_core(logits, temp, k, p, keys):
    """Everything that touches the whole vocab, in one compiled program.

    Run eagerly this is ~40 ms at batch 64 -- more than a decode step -- and
    essentially all of it is per-op dispatch, not arithmetic.
    """
    V = logits.shape[1]
    greedy = temp == 0
    out = logits / jnp.where(greedy, 1.0, temp)[:, None]
    out = apply_top_k(out, k)
    out = apply_top_p(out, p)

    # Gumbel-max: argmax(logits + Gumbel noise) is an exact categorical draw,
    # and it vmaps where a per-row categorical with per-row keys does not.
    gumbel = jax.vmap(lambda key: jax.random.gumbel(key, (V,), jnp.float32))(keys)
    sampled = jnp.argmax(out + gumbel, axis=-1)
    return jnp.where(greedy, jnp.argmax(out, axis=-1), sampled).astype(jnp.int32)


def sample(logits, params, prev_tokens=None, key=None):
    """logits (B, V) -> (B,) token ids. One vectorized pass over the batch."""
    out = logits.astype(jnp.float32)
    if prev_tokens is not None:
        pen = jnp.asarray([p.repetition_penalty for p in params])
        out = apply_repetition_penalty(out, prev_tokens, pen)

    return _sample_core(
        out,
        jnp.asarray([p.temperature for p in params], jnp.float32),
        jnp.asarray([p.top_k for p in params], jnp.int32),
        jnp.asarray([p.top_p for p in params], jnp.float32),
        row_keys(params, key),
    )
