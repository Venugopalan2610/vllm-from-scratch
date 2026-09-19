"""Reference solution, stage 13 (jax) - batched sampler."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

# A seeded row folds its seed into a FIXED base key. So the key of a request
# depends only on its seed, and nothing else in the batch can change it.
SEED_BASE_KEY = jax.random.key(0)


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: int = 0            # 0 = off
    top_p: float = 1.0        # 1.0 = off
    repetition_penalty: float = 1.0
    seed: int | None = None

    @property
    def greedy(self):
        return self.temperature == 0.0


def apply_repetition_penalty(logits, prev_tokens, penalties):
    """logits (rows, vocab), prev_tokens list[list[int]], penalties (rows,).

    The rows are ragged, so this function stays outside jit. That is
    acceptable: it touches a few indices in each row, not the vocabulary.
    """
    for row, tokens in enumerate(prev_tokens):
        penalty = float(penalties[row])
        if not tokens or penalty == 1.0:
            continue
        seen = jnp.asarray(sorted(set(tokens)), dtype=jnp.int32)
        seen_logits = logits[row, seen]
        # Divide a positive logit, MULTIPLY a negative one. A division makes
        # a negative logit larger, and that rewards the token.
        logits = logits.at[row, seen].set(jnp.where(
            seen_logits > 0, seen_logits / penalty, seen_logits * penalty))
    return logits


def apply_top_k(logits, k):
    """k (rows,). A 0 turns top-k off for that row.

    Do not return early on `int(k.max()) <= 0`. That reads a device value on
    the host, which is a sync, and jit cannot trace it. The `off` term does
    the same work at no cost.
    """
    descending = jnp.sort(logits, axis=-1)[:, ::-1]    # jnp.sort ascends
    kept_rank = jnp.clip(k, 1, logits.shape[1]) - 1
    threshold = jnp.take_along_axis(descending, kept_rank[:, None], axis=1)
    off = (k <= 0)[:, None]
    return jnp.where(off | (logits >= threshold), logits, -jnp.inf)


def apply_top_p(logits, p):
    """Keep the smallest set of tokens whose cumulative probability reaches
    p. Always keep the top token."""
    probs = jax.nn.softmax(logits, axis=-1)
    order = jnp.argsort(probs, axis=-1)[:, ::-1]
    sorted_probs = jnp.take_along_axis(probs, order, axis=1)
    mass_before = jnp.cumsum(sorted_probs, axis=-1) - sorted_probs
    # A token is not necessary if the mass before it already reaches p.
    remove_sorted = (mass_before >= p[:, None] - 1e-9).at[:, 0].set(False)
    rows = jnp.arange(logits.shape[0])[:, None]
    remove = jnp.zeros_like(remove_sorted).at[rows, order].set(remove_sorted)
    return jnp.where(remove, -jnp.inf, logits)


def row_keys(params, key=None):
    """One PRNG key for each request: two vmapped fold_ins and a select.

    A Python loop over 64 requests is 64 device dispatches before the
    sampling starts. We measured it: that is most of the sampler time. A key
    is a value. Batch it like any other value.
    """
    unseeded_base = SEED_BASE_KEY if key is None else key
    seeds = jnp.asarray([-1 if p.seed is None else p.seed for p in params],
                        jnp.int32)
    rows = jnp.arange(len(params), dtype=jnp.int32)
    seeded = jax.vmap(lambda seed: jax.random.fold_in(SEED_BASE_KEY, seed))(
        seeds)
    unseeded = jax.vmap(lambda row: jax.random.fold_in(unseeded_base, row))(
        rows)
    key_data = jnp.where((seeds >= 0)[:, None], jax.random.key_data(seeded),
                         jax.random.key_data(unseeded))
    return jax.random.wrap_key_data(key_data)


@jax.jit
def _sample_compiled(logits, temperature, top_k, top_p, keys):
    """All the work on the whole vocabulary, in one compiled program.

    Without jit this takes about 40 ms at batch 64, more than a decode step.
    Almost all of it is the dispatch of each op, not arithmetic.
    """
    greedy = temperature == 0
    scores = logits / jnp.where(greedy, 1.0, temperature)[:, None]
    scores = apply_top_p(apply_top_k(scores, top_k), top_p)
    # Gumbel-max: argmax(logits + Gumbel noise) is an exact categorical
    # draw. It vmaps. A categorical with a key for each row does not.
    vocab_size = logits.shape[1]
    noise = jax.vmap(lambda row_key: jax.random.gumbel(
        row_key, (vocab_size,), jnp.float32))(keys)
    sampled = jnp.argmax(scores + noise, axis=-1)
    return jnp.where(greedy, jnp.argmax(scores, axis=-1),
                     sampled).astype(jnp.int32)


def row_values(params, field, dtype=jnp.float32):
    return jnp.asarray([getattr(p, field) for p in params], dtype)


def sample(logits, params, prev_tokens=None, key=None):
    """logits (rows, vocab) -> (rows,) token ids, in one vectorized pass."""
    scores = logits.astype(jnp.float32)
    if prev_tokens is not None:
        scores = apply_repetition_penalty(
            scores, prev_tokens, row_values(params, "repetition_penalty"))
    return _sample_compiled(scores, row_values(params, "temperature"),
                            row_values(params, "top_k", jnp.int32),
                            row_values(params, "top_p"),
                            row_keys(params, key))
